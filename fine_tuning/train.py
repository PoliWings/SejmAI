import argparse
import json
import os
import logging
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from peft.utils import prepare_model_for_kbit_training

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
args = parser.parse_args()

DATASET_PATH = args.data_path
BASE_MODEL = "speakleash/Bielik-11B-v2.2-Instruct"
OUTPUT_DIR = f"./output/{DATASET_PATH.split('/')[-1].split('.')[0]}"
MAX_LENGTH = 16
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
LEARNING_RATE = 1e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.0
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EVAL_STEPS = 1
SAVE_STEPS = 1
LOGGING_STEPS = 1

# Load dataset
with open(DATASET_PATH, 'r') as f:
    data = json.load(f)

train_data = []
for record in data['train']:
    messages = record['messages']
    text = ""
    for msg in messages:
        text += f"{msg['role']}: {msg['content']}\n"
    train_data.append({'text': text.strip()})

val_data = []
for record in data['validation']:
    messages = record['messages']
    text = ""
    for msg in messages:
        text += f"{msg['role']}: {msg['content']}\n"
    val_data.append({'text': text.strip()})

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    result = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    result["labels"] = result["input_ids"].copy()
    return result

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Load model
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", load_in_4bit=True)
model = prepare_model_for_kbit_training(model)

# Lora configuration
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply PEFT (Low-rank adaptation) to the model
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=1,
    report_to=[],  # Disable reporting to external services
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Check if checkpoint exists, if not, train from scratch
checkpoint = None

# Check if a checkpoint exists in the directory
if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
    checkpoint_folders = [folder for folder in os.listdir(OUTPUT_DIR) if folder.startswith('checkpoint-')]
    if checkpoint_folders:
        latest_checkpoint = max(checkpoint_folders, key=lambda x: int(x.split('-')[1]))  # Get the folder with the highest number
        checkpoint = os.path.join(OUTPUT_DIR, latest_checkpoint)
        logger.info(f"Resuming from checkpoint at {checkpoint}")
else:
    logger.info(f"No checkpoint found, starting from scratch.")


# Train the model
trainer.train(resume_from_checkpoint=checkpoint)

# Save the model and tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.info(f"Training complete. Model saved to {OUTPUT_DIR}")
