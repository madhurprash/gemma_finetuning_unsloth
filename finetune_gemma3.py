"""
Fine-tune Gemma 3 270M model using Unsloth.

This script replicates the functionality of the Unsloth Gemma3 (270M) notebook.
It loads the model, prepares the dataset, and fine-tunes using LoRA adapters.
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


# Constants
MAX_SEQ_LENGTH: int = 2048
MODEL_NAME: str = "unsloth/gemma-3-270m-it"
DATASET_NAME: str = "Thytu/ChessInstruct"
DATASET_SPLIT: str = "train[:10000]"
OUTPUT_DIR: str = "outputs"

# LoRA Configuration
LORA_R: int = 128
LORA_ALPHA: int = 128
LORA_DROPOUT: float = 0
TARGET_MODULES: list = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Training Configuration
BATCH_SIZE: int = 8
GRADIENT_ACCUMULATION_STEPS: int = 1
WARMUP_STEPS: int = 5
MAX_STEPS: int = 100
LEARNING_RATE: float = 5e-5
WEIGHT_DECAY: float = 0.01
SEED: int = 3407


def _load_model_and_tokenizer():
    """Load the Gemma 3 270M model and tokenizer."""
    logger.info(f"Loading model: {MODEL_NAME}")

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )

    logger.info("Model loaded successfully")
    return model, tokenizer


def _get_peft_model(model):
    """Apply LoRA adapters to the model."""
    logger.info("Applying LoRA adapters")

    model = FastModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    logger.info("LoRA adapters applied successfully")
    return model


def _convert_to_chatml(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert dataset example to ChatML format."""
    return {
        "conversations": [
            {"role": "system", "content": example["task"]},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["expected_output"]},
        ]
    }


def _load_and_prepare_dataset(tokenizer):
    """Load and prepare the dataset."""
    logger.info(f"Loading dataset: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    logger.info(f"Loaded {len(dataset)} examples")

    # Convert to ChatML format
    logger.info("Converting dataset to ChatML format")
    dataset = dataset.map(_convert_to_chatml)

    # Format prompts using chat template
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            ).removeprefix("<bos>")
            for convo in convos
        ]
        return {"text": texts}

    logger.info("Formatting prompts with chat template")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    logger.info("Dataset prepared successfully")
    return dataset


def _create_trainer(
    model,
    tokenizer,
    dataset,
):
    """Create SFT trainer with configuration."""
    logger.info("Creating SFT trainer")

    training_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear",
        seed=SEED,
        output_dir=OUTPUT_DIR,
        report_to="none",
    )

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
    """Run inference on a sample from the dataset."""
    logger.info("Running inference example")

    # Use the 11th example from the dataset (index 10)
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

    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate response
    from transformers import TextStreamer

    text_streamer = TextStreamer(tokenizer)

    logger.info("Generating response...")
    _ = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
    )


def main():
    """Main function to orchestrate the fine-tuning process."""
    logger.info("Starting Gemma 3 270M fine-tuning")

    # Load model and tokenizer
    model, tokenizer = _load_model_and_tokenizer()

    # Apply LoRA adapters
    model = _get_peft_model(model)

    # Load and prepare dataset
    dataset = _load_and_prepare_dataset(tokenizer)

    # Create trainer
    trainer = _create_trainer(model, tokenizer, dataset)

    # Train the model
    logger.info("Starting training...")
    trainer_stats = trainer.train()
    logger.info("Training completed")
    logger.info(f"Training stats: {trainer_stats}")

    # Run inference example
    _run_inference(model, tokenizer, dataset)

    # Save the model
    logger.info(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Model saved successfully")

    logger.info("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
