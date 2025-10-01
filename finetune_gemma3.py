"""
Fine-tune Gemma 3 270M model using Unsloth.

This script demonstrates how to fine-tune a small language model using Unsloth,
a library that provides 2x faster training with 70% less VRAM compared to standard
approaches. Unsloth achieves this through optimized Triton kernels and manual
backpropagation derivations.

Key Features of Unsloth:
- 2x faster training speed than standard Transformers
- 60-70% reduction in memory usage
- Full compatibility with Hugging Face ecosystem (transformers, PEFT, TRL)
- Support for LoRA and QLoRA (quantized LoRA) training
- Zero accuracy loss compared to standard training methods
"""

import logging
from typing import Any, Dict

from datasets import load_dataset
from unsloth import FastModel
from trl import SFTConfig, SFTTrainer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# ===========================
# MODEL AND DATASET CONSTANTS
# ===========================
MAX_SEQ_LENGTH: int = 2048  # Maximum sequence length for tokenization
MODEL_NAME: str = "unsloth/gemma-3-270m-it"  # Pre-optimized Gemma 3 model from Unsloth
DATASET_NAME: str = "Thytu/ChessInstruct"  # Chess instruction dataset
DATASET_SPLIT: str = "train[:10000]"  # Use first 10,000 training examples
OUTPUT_DIR: str = "outputs"  # Directory to save the fine-tuned model


# ==================
# LoRA CONFIGURATION
# ==================
# LoRA (Low-Rank Adaptation) fine-tunes models by adding small trainable adapter
# matrices to existing layers, drastically reducing trainable parameters while
# maintaining quality. Instead of updating all model weights, LoRA injects
# low-rank matrices that capture task-specific adaptations.

LORA_R: int = 128  # Rank of LoRA adapters (controls adapter capacity)
                   # Higher rank = more parameters = potentially better accuracy
                   # but also more memory usage. Common values: 8, 16, 32, 64, 128

LORA_ALPHA: int = 128  # Scaling factor for LoRA updates
                       # Controls the strength of fine-tuned adjustments
                       # Common practice: set equal to LORA_R or 2*LORA_R
                       # Formula: effective_learning_rate = alpha/r * base_lr

LORA_DROPOUT: float = 0  # Dropout rate for LoRA layers
                         # Set to 0 for Unsloth optimization (faster training)
                         # Non-zero values can help prevent overfitting

TARGET_MODULES: list = [
    # Apply LoRA to all major linear layers for best results
    # Research shows targeting all layers matches full fine-tuning performance
    "q_proj",      # Query projection in attention
    "k_proj",      # Key projection in attention
    "v_proj",      # Value projection in attention
    "o_proj",      # Output projection in attention
    "gate_proj",   # Gate projection in MLP (feed-forward network)
    "up_proj",     # Up projection in MLP
    "down_proj",   # Down projection in MLP
]


# ======================
# TRAINING CONFIGURATION
# ======================
BATCH_SIZE: int = 8  # Number of examples per training step per device
                     # Larger batch = more stable gradients but more memory

GRADIENT_ACCUMULATION_STEPS: int = 1  # Accumulate gradients over N steps
                                      # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
                                      # Useful for simulating larger batches with limited memory

WARMUP_STEPS: int = 5  # Linear warmup for learning rate
                       # Gradually increases LR from 0 to target over N steps
                       # Helps stabilize training at the start

MAX_STEPS: int = 100  # Total number of training steps
                      # Training will stop after this many optimization steps

LEARNING_RATE: float = 5e-5  # Learning rate for optimizer
                             # Controls how much to update weights each step
                             # Typical range for fine-tuning: 1e-5 to 5e-5

WEIGHT_DECAY: float = 0.01  # L2 regularization coefficient
                            # Penalizes large weights to prevent overfitting
                            # Common values: 0.01 to 0.1

SEED: int = 3407  # Random seed for reproducibility
                  # Ensures consistent results across runs


def _load_model_and_tokenizer():
    """
    Load the Gemma 3 270M model and tokenizer using Unsloth's FastModel.
    
    Unsloth's FastModel.from_pretrained() is an optimized replacement for
    HuggingFace's AutoModel.from_pretrained(). It automatically patches the
    model with optimized Triton kernels for faster forward/backward passes.
    
    Returns:
        tuple: (model, tokenizer) - The loaded model and its tokenizer
    
    Unsloth Optimization Details:
    - Replaces attention mechanisms with Flash Attention variants
    - Optimizes RoPE (Rotary Position Embeddings) computations
    - Implements custom autograd functions for efficient backpropagation
    - Reduces memory fragmentation during training
    """
    logger.info(f"Loading model: {MODEL_NAME}")
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,  # Maximum context window
        load_in_4bit=False,  # Disable 4-bit quantization (QLoRA)
                            # Set to True for QLoRA: 4-bit model + LoRA adapters
                            # Reduces memory by ~75% with minimal accuracy loss
        load_in_8bit=False,  # Disable 8-bit quantization
                            # 8-bit uses 2x memory vs 4-bit but slightly more accurate
        full_finetuning=False,  # Use LoRA instead of full fine-tuning
                               # Full fine-tuning updates all parameters (more memory/time)
                               # LoRA only updates small adapter matrices (efficient)
    )

    logger.info("Model loaded successfully")
    return model, tokenizer


def _get_peft_model(model):
    """
    Apply LoRA (Low-Rank Adaptation) adapters to the model using Unsloth.
    
    This function uses FastModel.get_peft_model() to inject trainable LoRA
    adapter matrices into the specified target modules. PEFT (Parameter-Efficient
    Fine-Tuning) allows fine-tuning with a fraction of trainable parameters.
    
    Args:
        model: The base model to add LoRA adapters to
    
    Returns:
        model: The model with LoRA adapters attached
    
    LoRA Theory:
    For a weight matrix W, LoRA represents updates as: W' = W + (alpha/r) * A * B
    where A and B are low-rank matrices with rank r, significantly reducing
    the number of trainable parameters from d×d to d×r + r×d.
    
    Unsloth Advantage:
    - Unsloth's implementation is optimized for minimal memory overhead
    - Custom kernels fuse LoRA computation with attention operations
    - "unsloth" gradient checkpointing reduces memory by additional 30%
    """
    logger.info("Applying LoRA adapters")

    model = FastModel.get_peft_model(
        model,
        r=LORA_R,  # Rank of adapter matrices
        target_modules=TARGET_MODULES,  # Which layers to adapt
        lora_alpha=LORA_ALPHA,  # Scaling factor for adapter outputs
        lora_dropout=LORA_DROPOUT,  # Dropout for regularization
        bias="none",  # Don't train bias terms (saves memory, minimal impact)
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
                                              # Options: True, False, "unsloth"
                                              # "unsloth" provides 30% extra memory savings
                                              # Trades computation for memory by recomputing
                                              # activations during backward pass
        random_state=SEED,  # For reproducible initialization
    )

    logger.info("LoRA adapters applied successfully")
    return model


def _convert_to_chatml(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert dataset example to ChatML (Chat Markup Language) format.
    
    ChatML is a standardized format for representing conversations with roles:
    - system: Instructions/context for the model
    - user: The user's input/question
    - assistant: The model's expected response
    
    Args:
        example: Dictionary with 'task', 'input', and 'expected_output' keys
    
    Returns:
        Dictionary with 'conversations' key containing role-based messages
    
    This conversion ensures the model learns the correct conversational structure
    and can distinguish between different message types during training.
    """
    return {
        "conversations": [
            {"role": "system", "content": example["task"]},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["expected_output"]},
        ]
    }


def _load_and_prepare_dataset(tokenizer):
    """
    Load and prepare the chess instruction dataset for training.
    
    This function:
    1. Loads the dataset from HuggingFace Hub
    2. Converts to ChatML format for proper role-based training
    3. Applies the chat template to format prompts correctly
    4. Creates the 'text' field that SFTTrainer expects
    
    Args:
        tokenizer: The tokenizer with chat template
    
    Returns:
        dataset: Processed dataset ready for training
    
    Dataset Processing:
    - Chat templates ensure consistent formatting across different models
    - The template wraps conversations with special tokens (e.g., <bos>, <eos>)
    - Removing <bos> prefix prevents duplication (tokenizer adds it)
    """
    logger.info(f"Loading dataset: {DATASET_NAME}")

    # Load subset of chess instruction dataset
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    logger.info(f"Loaded {len(dataset)} examples")

    # Convert each example to ChatML format
    logger.info("Converting dataset to ChatML format")
    dataset = dataset.map(_convert_to_chatml)

    # Format prompts using the model's chat template
    def formatting_prompts_func(examples):
        """
        Apply chat template to format conversations into text strings.
        
        The chat template converts structured conversations into the model's
        expected format with proper special tokens and formatting.
        """
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,  # Return string, not token IDs
                add_generation_prompt=False,  # Don't add prompt for generation
            ).removeprefix("<bos>")  # Remove <bos> token (added automatically)
            for convo in convos
        ]
        return {"text": texts}

    logger.info("Formatting prompts with chat template")
    # Apply formatting in batches for efficiency
    dataset = dataset.map(formatting_prompts_func, batched=True)

    logger.info("Dataset prepared successfully")
    return dataset


def _create_trainer(
    model,
    tokenizer,
    dataset,
):
    """
    Create SFTTrainer (Supervised Fine-Tuning Trainer) with configuration.
    
    SFTTrainer is from the TRL (Transformers Reinforcement Learning) library
    and simplifies supervised fine-tuning. It's the standard first step in RLHF
    (Reinforcement Learning from Human Feedback) pipelines.
    
    Args:
        model: The model with LoRA adapters
        tokenizer: The tokenizer for the model
        dataset: The prepared training dataset
    
    Returns:
        trainer: Configured SFTTrainer ready for training
    
    Training Details:
    - Uses AdamW optimizer with 8-bit precision for memory efficiency
    - Linear learning rate schedule with warmup for stable convergence
    - Logs metrics every step for monitoring
    """
    logger.info("Creating SFT trainer")

    # Configure training arguments
    training_args = SFTConfig(
        dataset_text_field="text",  # Field containing formatted text
        per_device_train_batch_size=BATCH_SIZE,  # Batch size per GPU
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # Steps before update
        warmup_steps=WARMUP_STEPS,  # Linear warmup period
        max_steps=MAX_STEPS,  # Total training steps
        learning_rate=LEARNING_RATE,  # Optimizer learning rate
        logging_steps=1,  # Log every step
        optim="adamw_8bit",  # 8-bit Adam optimizer for memory efficiency
                            # Uses less memory than standard adamw_torch
                            # Minimal impact on convergence quality
        weight_decay=WEIGHT_DECAY,  # L2 regularization strength
        lr_scheduler_type="linear",  # Linear decay from peak to 0
                                    # Other options: cosine, constant, etc.
        seed=SEED,  # For reproducible training
        output_dir=OUTPUT_DIR,  # Where to save checkpoints
        report_to="none",  # Disable logging to WandB/TensorBoard
                          # Set to "wandb" or "tensorboard" for experiment tracking
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    logger.info("Trainer created successfully")
    return trainer


def _run_inference(
    model,
    tokenizer,
    dataset,
):
    """
    Run inference on a sample from the dataset to test the fine-tuned model.
    
    This demonstrates how to use the model for generation after training.
    The example uses streaming output to show tokens as they're generated,
    which is useful for interactive applications.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        dataset: The dataset (to get a test example)
    
    Generation Strategy:
    - Uses the 11th example (index 10) from the dataset
    - Formats as a conversation with system and user messages
    - Generates up to 128 new tokens
    - Streams output token-by-token for real-time display
    """
    logger.info("Running inference example")

    # Select the 11th example from the dataset
    sample_idx = 10
    messages = [
        {
            "role": "system",
            "content": dataset["conversations"][sample_idx][0]["content"],
        },
        {
            "role": "user",
            "content": dataset["conversations"][sample_idx][1]["content"],
        },
    ]

    # Apply chat template and tokenize
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,  # Return token IDs, not strings
        add_generation_prompt=True,  # Add prompt template for generation
        return_tensors="pt",  # Return PyTorch tensors
    ).to("cuda")  # Move to GPU

    # Generate response with streaming output
    from transformers import TextStreamer

    # TextStreamer prints tokens as they're generated
    text_streamer = TextStreamer(tokenizer)

    logger.info("Generating response...")
    _ = model.generate(
        input_ids=inputs,
        streamer=text_streamer,  # Stream tokens to console
        max_new_tokens=128,  # Maximum tokens to generate
        use_cache=True,  # Cache key-value pairs for faster generation
                        # Significantly speeds up autoregressive generation
    )


def main():
    """
    Main function to orchestrate the complete fine-tuning pipeline.
    
    Pipeline Steps:
    1. Load pre-trained Gemma 3 model and tokenizer
    2. Apply LoRA adapters for parameter-efficient training
    3. Load and format the chess instruction dataset
    4. Configure the supervised fine-tuning trainer
    5. Execute the training loop
    6. Test the model with inference
    7. Save the fine-tuned model and tokenizer
    
    The entire process takes advantage of Unsloth's optimizations:
    - Faster training through optimized kernels
    - Reduced memory usage through efficient implementations
    - Seamless integration with HuggingFace ecosystem
    """
    logger.info("Starting Gemma 3 270M fine-tuning with Unsloth")

    # Step 1: Load base model and tokenizer
    # Unsloth's FastModel automatically applies speed optimizations
    model, tokenizer = _load_model_and_tokenizer()

    # Step 2: Apply LoRA adapters
    # Only the adapter weights will be trained, not the base model
    model = _get_peft_model(model)

    # Step 3: Load and prepare dataset
    # Convert to ChatML format and apply chat template
    dataset = _load_and_prepare_dataset(tokenizer)

    # Step 4: Create trainer
    # SFTTrainer handles the training loop, logging, and checkpointing
    trainer = _create_trainer(model, tokenizer, dataset)

    # Step 5: Train the model
    # This is where the actual fine-tuning happens
    logger.info("Starting training...")
    trainer_stats = trainer.train()
    logger.info("Training completed")
    logger.info(f"Training stats: {trainer_stats}")

    # Step 6: Run inference example
    # Test the model on a sample to verify it learned correctly
    _run_inference(model, tokenizer, dataset)

    # Step 7: Save the model
    # Saves LoRA adapter weights (much smaller than full model)
    logger.info(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)  # Saves adapter weights
    tokenizer.save_pretrained(OUTPUT_DIR)  # Saves tokenizer config
    logger.info("Model saved successfully")

    logger.info("Fine-tuning completed successfully!")
    logger.info("The saved model can be loaded later and merged with the base model")
    logger.info("or used directly with PEFT for inference with minimal memory overhead")


if __name__ == "__main__":
    main()
