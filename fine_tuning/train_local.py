import argparse
import json
import os
import datetime
import logging
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from dotenv import load_dotenv

load_dotenv("../.env")

os.environ["WANDB_PROJECT"] = "local_training"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, required=True, help="Path to training dataset (JSON)")
parser.add_argument("--base-model", type=str, required=True, help="Base model name or path (e.g., Hugging Face hub ID)")
args = parser.parse_args()

DATASET_PATH = args.data_path
BASE_MODEL = args.base_model

base_name = BASE_MODEL.replace("/", "_")
dataset_name = os.path.splitext(os.path.basename(DATASET_PATH))[0]
OUTPUT_DIR = f"./output/{base_name}__{dataset_name}"

# Hyperparameters
MAX_LENGTH = 2048
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.0
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
SAVE_STEPS = 100
EVAL_STEPS = 100
LOGGING_STEPS = 10
SEED = 42

# Load raw JSON data
with open(DATASET_PATH, "r") as f:
    data = json.load(f)

# Prepare train/validation lists
train_data = []
for rec in data["train"]:
    text = "".join([f"{m['role']}: {m['content']}\n" for m in rec["messages"]])
    train_data.append({"text": text.strip()})

val_data = []
for rec in data["validation"]:
    text = "".join([f"{m['role']}: {m['content']}\n" for m in rec["messages"]])
    val_data.append({"text": text.strip()})

# Build Hugging Face Datasets
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Tokenization function
def tokenize_fn(examples):
    out = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    out["labels"] = out["input_ids"].copy()
    return out


# Tokenize datasets
train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# Load and prepare model for k-bit training
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", load_in_4bit=True)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

# Apply LoRA PEFT
lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# SFT Trainer config
sft_config = SFTConfig(
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    optim="adamw_torch",
    dataset_text_field="text",
    max_seq_length=MAX_LENGTH,
    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    save_total_limit=1,
    output_dir=OUTPUT_DIR,
    seed=SEED,
    report_to="wandb",
    run_name=f"{base_name}__{dataset_name}__{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    push_to_hub=False,
)

# Initialize SFT Trainer
tool_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Check for existing checkpoints
if os.path.isdir(OUTPUT_DIR):
    ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if ckpts:
        latest = max(ckpts, key=lambda x: int(x.split("-")[-1]))
        res_path = os.path.join(OUTPUT_DIR, latest)
        logger.info(f"Resuming from checkpoint {res_path}")
        tool_trainer.train(resume_from_checkpoint=res_path)
    else:
        logger.info("No checkpoint found, training from scratch.")
        tool_trainer.train()
else:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tool_trainer.train()

# Save final model/tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
logger.info(f"Training complete. Model saved in {OUTPUT_DIR}")
