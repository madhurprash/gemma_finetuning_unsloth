# Gemma 3 270M Fine-tuning with Unsloth

Fine-tune Google's Gemma 3 270M instruction-tuned model using Unsloth on AWS SageMaker with a chess instruction dataset.

## Overview

This project demonstrates efficient fine-tuning of small language models using Unsloth, a library that provides 2x faster training with 60-70% less VRAM compared to standard approaches. The fine-tuning is performed on AWS SageMaker using a `ml.g5.2xlarge` instance.

### Key Benefits of Unsloth

- **2x faster training** than standard Transformers library
- **60-70% memory reduction** through optimized Triton kernels
- **Zero accuracy loss** compared to standard training methods
- **Full compatibility** with Hugging Face ecosystem (transformers, PEFT, TRL)
- **LoRA and QLoRA support** for parameter-efficient fine-tuning

## Prerequisites

- Python 3.11+
- AWS SageMaker with `ml.g5.2xlarge` instance (NVIDIA A10G GPU, 24GB GPU memory)
- Access to Hugging Face Hub for model and dataset downloads

## Installation

Install dependencies using `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv && source .venv/bin/activate && uv pip sync pyproject.toml
UV_PROJECT_ENVIRONMENT=.venv
uv add zmq
python -m ipykernel install --user --name=.venv --display-name="Python (uv env)"
```

### Run the fine-tuning job

```python
python finetune_gemma3.py
```

## Model and Dataset

### Model
- **Base Model**: `unsloth/gemma-3-270m-it` (pre-optimized Gemma 3 from Unsloth)
- **Architecture**: Gemma 3 270M parameters (instruction-tuned variant)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Max Sequence Length**: 2048 tokens

### Dataset
- **Dataset**: [Thytu/ChessInstruct](https://huggingface.co/datasets/Thytu/ChessInstruct)
- **Split**: First 10,000 training examples
- **Format**: Task-based instruction format with system prompts, user inputs, and expected outputs

## Configuration

### LoRA Configuration

The project uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning:

```python
LORA_R = 128              # Rank of LoRA adapters
LORA_ALPHA = 128          # Scaling factor for LoRA updates
LORA_DROPOUT = 0          # Dropout disabled for Unsloth optimization
```

**Target Modules**: All major transformer layers are adapted:
- Query, Key, Value, Output projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- MLP layers (`gate_proj`, `up_proj`, `down_proj`)

### Training Hyperparameters

```python
BATCH_SIZE = 8                      # Batch size per device
GRADIENT_ACCUMULATION_STEPS = 1     # Effective batch size = 8
WARMUP_STEPS = 5                    # Linear warmup steps
MAX_STEPS = 100                     # Total training steps
LEARNING_RATE = 5e-5                # AdamW learning rate
WEIGHT_DECAY = 0.01                 # L2 regularization
```

## Usage

### Basic Training

Run the fine-tuning script:

```bash
uv run python finetune_gemma3.py
```

### Training Pipeline

The script executes the following steps:

1. **Load Model**: Downloads and loads the pre-optimized Gemma 3 270M model
2. **Apply LoRA Adapters**: Attaches parameter-efficient LoRA adapters to the model
3. **Load Dataset**: Downloads and processes the ChessInstruct dataset
4. **Format Data**: Converts examples to ChatML format with system/user/assistant roles
5. **Train Model**: Runs supervised fine-tuning for 100 steps
6. **Test Inference**: Generates a sample output to verify the model
7. **Save Model**: Saves LoRA adapter weights to `outputs/` directory

### Expected Training Time

On `ml.g5.2xlarge` (NVIDIA A10G, 24GB GPU):
- **Training**: ~3-5 minutes for 100 steps with batch size 8
- **Memory Usage**: ~8-10GB GPU memory (leaves headroom for larger batches)

## Output

The fine-tuned model is saved to the `outputs/` directory

**Note**: Only LoRA adapter weights are saved (typically 50-100MB), not the full model (540MB).

## Technical Details

### LoRA (Low-Rank Adaptation)

LoRA fine-tunes models by adding small trainable adapter matrices to existing layers. For a weight matrix W, LoRA represents updates as:


### Unsloth Optimizations

Unsloth achieves 2x speedup and 60-70% memory reduction through:

1. **Flash Attention Variants**: Optimized attention mechanisms
2. **Custom Triton Kernels**: Hand-optimized GPU kernels for RoPE embeddings
3. **Manual Backpropagation**: Custom autograd functions for efficient gradients
4. **Gradient Checkpointing**: "unsloth" mode provides 30% extra memory savings
5. **8-bit AdamW Optimizer**: Reduces optimizer memory footprint

### ChatML Format

The training data is converted to ChatML (Chat Markup Language) format:

```json
{
  "conversations": [
    {"role": "system", "content": "Task description"},
    {"role": "user", "content": "User input"},
    {"role": "assistant", "content": "Expected output"}
  ]
}
```

This ensures the model learns proper conversational structure and role distinctions.

## Loading the Fine-tuned Model

### Option 1: Load with PEFT (Memory Efficient)

```python
from unsloth import FastModel
from peft import PeftModel

# Load base model
model, tokenizer = FastModel.from_pretrained("unsloth/gemma-3-270m-it")

# Load LoRA adapters
model = PeftModel.from_pretrained(model, "outputs/")

# Run inference
messages = [
    {"role": "system", "content": "Chess instruction system"},
    {"role": "user", "content": "Your question here"}
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

### Option 2: Merge and Save Full Model

```python
from unsloth import FastModel

# Load and merge
model, tokenizer = FastModel.from_pretrained("unsloth/gemma-3-270m-it")
model = PeftModel.from_pretrained(model, "outputs/")
model = model.merge_and_unload()  # Merge adapters into base model

# Save full model
model.save_pretrained("merged_model/")
tokenizer.save_pretrained("merged_model/")
```

## Monitoring and Logging

The script logs progress at each step:
- Model loading status
- Dataset processing progress
- Training loss per step
- Inference examples
- Model saving confirmation

All logs use Python's `logging` module with timestamps and file locations.

## AWS SageMaker Instance Details

### ml.g5.2xlarge Specifications

- **GPU**: 1x NVIDIA A10G Tensor Core GPU
- **GPU Memory**: 24GB GDDR6
- **vCPUs**: 8
- **System RAM**: 32GB
- **Network**: Up to 10 Gbps
- **Storage**: EBS-optimized

This instance provides sufficient GPU memory for:
- Training with batch size 8-16
- LoRA rank up to 256
- Full attention computation without gradient checkpointing
- Concurrent training and monitoring

## Customization

### Adjust Training Steps

Modify `MAX_STEPS` in `finetune_gemma3.py`:

```python
MAX_STEPS: int = 500  # Train for more steps
```

### Enable QLoRA (4-bit Quantization)

For 75% memory reduction with minimal accuracy loss:

```python
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    load_in_4bit=True,  # Enable 4-bit quantization
    # ... other parameters
)
```

### Use Different Dataset

Replace with your own instruction dataset:

```python
DATASET_NAME: str = "your-username/your-dataset"
DATASET_SPLIT: str = "train"
```

Ensure your dataset has `task`, `input`, and `expected_output` fields, or modify `_convert_to_chatml()` accordingly.


## References

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Gemma Model Card](https://huggingface.co/google/gemma-3-270m-it)
- [ChessInstruct Dataset](https://huggingface.co/datasets/Thytu/ChessInstruct)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
