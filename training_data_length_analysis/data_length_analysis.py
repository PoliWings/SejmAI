from transformers import AutoTokenizer
from datasets import Dataset
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

BASE_MODEL = "speakleash/Bielik-11B-v2.2-Instruct"
DATASET_PATH = args.data_path

filename = DATASET_PATH.split("/")[-1]
if filename.endswith(".json"):
    filename = filename[:-5]

# Load raw JSON data
with open(DATASET_PATH, "r", encoding="utf-8") as f:
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
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token="")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Tokenization function
def tokenize_fn(examples):
    out = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
    )
    out["labels"] = out["input_ids"].copy()
    return out


# Tokenize datasets
train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

pad_token_id = tokenizer.pad_token_id

# Calculate token lengths for the entire dataset
train_lengths = [len(example["input_ids"]) for example in train_dataset]

# Calculate statistical insights for the train set
train_min_len = np.min(train_lengths)
train_max_len = np.max(train_lengths)
train_avg_len = np.mean(train_lengths)
train_median_len = np.median(train_lengths)
train_std_len = np.std(train_lengths)
train_90th_percentile = np.percentile(train_lengths, 90)

# Print out the statistics
print(f"Training Set Token Lengths Statistics:")
print(
    f"Min: {train_min_len}, Max: {train_max_len}, Average: {train_avg_len}, Median: {train_median_len}, Standard Deviation: {train_std_len}, 90th percentile: {train_90th_percentile}"
)

# Plot histogram for the training dataset token lengths
plt.figure(figsize=(12, 6))

# Plot for train dataset
plt.hist(
    train_lengths,
    bins=100,
    color="deepskyblue",
    edgecolor="black",
    linewidth=0.5,
    label=f"Min: {train_min_len}, Max: {train_max_len}, Std: {train_std_len:.2f}",
)
plt.title(f'Histogram of Token Lengths (Training data {filename.split("_")[0]})')
plt.xlabel("Token Length")
plt.ylabel("Frequency")
plt.axvline(
    train_90th_percentile,
    color="red",
    linestyle="dashed",
    linewidth=1,
    label=f"90th Percentile: {train_90th_percentile:.2f}",
)
plt.axvline(
    train_avg_len,
    color="navy",
    linestyle="dashed",
    linewidth=1,
    label=f"Average: {train_avg_len:.2f}",
)
plt.axvline(
    train_median_len,
    color="magenta",
    linestyle="dashed",
    linewidth=1,
    label=f"Median: {train_median_len:.2f}",
)
plt.legend()

plt.tight_layout()
plt.savefig(f"training_data_histogram_{filename}.png")
plt.close()
