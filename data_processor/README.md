# Data Processor

Transforms raw scraped parliamentary speeches into training-ready datasets for LLM fine-tuning. Handles MP classification (left/right), text cleaning, and dataset generation in **SFT** and **DPO** formats.

## Overview

| Script                | Purpose                                                                    |
| --------------------- | -------------------------------------------------------------------------- |
| `map_members.py`      | Classifies MPs as left-wing or right-wing based on club affiliation        |
| `process_data.py`     | Cleans speeches and formats them into SFT/DPO training datasets            |
| `llm_connection.py`   | Helper module for LLM-based context generation (used by `process_data.py`) |
| `chat_template.ipynb` | Notebook demonstrating model loading and chat template formatting          |

## Setup

```bash
pip install -r requirements.txt
```

It is recommended to use a virtual environment (venv or conda).

## Usage

### Step 1: Classify MPs

First, extract all parliamentary clubs from the scraped data:

```bash
python map_members.py --get_clubs
```

This generates `club_mapping.json`. Manually fill in each club's alignment as `"left"`, `"right"`, or `""` (excluded).

Then map individual MPs based on the club mapping:

```bash
python map_members.py --get_members
```

This generates `member_mapping.json` with each MP's political classification.

### Step 2: Generate datasets

```bash
python process_data.py
```

This produces SFT and DPO datasets split by political alignment (`left.json` and `right.json`).

### Step 2a: Generate missing contexts (optional)

Many scraped speeches lack a topic/context. To generate them using an LLM:

1. Create a `.env` file:

   ```
   LLM_URL=""
   LLM_USERNAME=""
   LLM_PASSWORD=""
   ```

2. Run with the `--gen_context` flag:
   ```bash
   python process_data.py --gen_context
   ```

The script saves checkpoints every 500 speeches for resumability.

## Output Structure

```
output/
├── sft/
│   ├── left.json
│   └── right.json
└── dpo/
    ├── left.json
    └── right.json
```

### SFT format

Standard conversational pairs where the `user` role contains the parliamentary topic and the `assistant` role contains the speech.

### DPO format

Prompt with `chosen` (politically aligned) and `rejected` (opposing alignment) response pairs.

## Data Pipeline

```
scraper output → map_members.py → process_data.py → training datasets
                     ↓                    ↓
              member_mapping.json    SFT & DPO JSON files
```

The processing pipeline performs the following:

- Filters speeches without political alignment
- Removes brackets, HTML artifacts, and formal greetings
- Adds synthetic context for speeches missing a topic (via LLM, optional)
- Formats into chat-style instruction-response pairs
