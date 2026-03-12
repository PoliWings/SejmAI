# Fine-Tuning

LoRA/QLoRA fine-tuning of LLMs on the Polish Parliamentary Speeches Corpus. Supports both **local training** (via Hugging Face Trainer) and **remote training** (via API service).

## Overview

| File               | Purpose                                                       |
| ------------------ | ------------------------------------------------------------- |
| `train_local.py`   | Local fine-tuning with LoRA on a CUDA GPU                     |
| `train_service.py` | Remote fine-tuning via LLM hosting API                        |
| `start.sh`         | Bash wrapper to train both left and right models sequentially |
| `convert.py`       | Converts SFT datasets into train/validation splits            |

## Setup

### 1. Check CUDA version

```bash
nvcc --version
```

If CUDA is not installed:

```bash
sudo apt install build-essential nvidia-cuda-toolkit -y
```

### 2. Create conda environment

```bash
conda create -n sejm python=3.10 -y && conda activate sejm
```

### 3. Install PyTorch

Install the version matching your CUDA. Example for CUDA 11.8:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install dependencies

```bash
pip install bitsandbytes datasets peft sentencepiece transformers ipykernel protobuf wandb huggingface_hub[cli] dotenv trl
```

### 5. Log in to Hugging Face

```bash
huggingface-cli login
```

## Usage

### Local training

```bash
python train_local.py --data-path <path_to_dataset.json> --base-model <model_id>
```

The trained adapter is saved to `./output/{model}__{dataset}/`.

### Batch training (left + right)

```bash
./start.sh --base-model <model_name>
```

Trains both left-wing and right-wing LoRA adapters sequentially.

### Remote training via API

Requires `.env` with `LLM_URL`, `LLM_USERNAME`, `LLM_PASSWORD`, and `WANDB_API_KEY`.

```bash
python train_service.py --start left    # Start left-wing training
python train_service.py --start right   # Start right-wing training
python train_service.py --status        # Check training status
python train_service.py --download-lora # Download trained adapter
```

## Training Hyperparameters

| Parameter           | Value                                                                                           |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| LoRA rank           | 32                                                                                              |
| LoRA alpha          | 32                                                                                              |
| LoRA dropout        | 0.1                                                                                             |
| LoRA targets        | All linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) |
| Batch size          | 1 per device (local) / 2 per device (remote)                                                    |
| Learning rate       | 1e-4 (cosine decay)                                                                             |
| Warm-up ratio       | 0.1                                                                                             |
| Max sequence length | 2048                                                                                            |
| Quantization        | 4-bit (QLoRA with bitsandbytes)                                                                 |
| Epochs              | 1                                                                                               |

## File Manager

Utility script for managing files on the remote hosting service. Requires `HOSTING_URL` in `.env`.

```bash
./file_manager.sh --help
```

## Useful Commands

```bash
# Run training in background
nohup ./start.sh --base-model <model_name> &> out.log &

# Monitor GPU & RAM usage
watch -n 0.5 -d "nvidia-smi && free -h"

# Follow training logs
tail -f out.log
```
