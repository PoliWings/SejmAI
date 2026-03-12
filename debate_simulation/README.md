# Debate Simulation

Simulates political debates between the left-wing and right-wing fine-tuned models. Both models take turns answering questions and responding to each other's arguments.

## Overview

| Script      | Purpose                                                               |
| ----------- | --------------------------------------------------------------------- |
| `debate.py` | Orchestrates multi-round debates between left and right LoRA adapters |

## How It Works

The debate proceeds in two phases:

1. **Moderator questions** — Both models answer predefined questions from a JSON file. Each model sees the other's response before formulating its own answer (randomized order).
2. **Model-generated questions** — Each model generates questions on random political topics and the opposing model responds.

The debate uses a TV debate persona system prompt — models are instructed to argue persuasively in Polish using a natural conversational tone, avoiding formal parliamentary language.

### Topic categories

Questions are drawn from 20 political domains including: foreign policy, economy, healthcare, education, national security, civil rights, climate, EU integration, and more.

## Setup

Requires the same conda environment as `fine_tuning/`. Create a `.env` file in the project root:

```
LLM_URL=""
LLM_USERNAME=""
LLM_PASSWORD=""
```

## Usage

### With predefined questions

```bash
python debate.py --service --questions <questions.json>
```

### With model-generated questions

```bash
python debate.py --service --ask-questions <number>
```

### Combined

```bash
python debate.py --service --questions <questions.json> --ask-questions 5
```

| Argument          | Description                                               |
| ----------------- | --------------------------------------------------------- |
| `--model-name`    | Hugging Face model ID (required if not using `--service`) |
| `--service`       | Use remote LLM API                                        |
| `--questions`     | Path to JSON file with predefined debate questions        |
| `--ask-questions` | Number of questions each model generates for the other    |

## Output

Results are saved to the `output/` directory:

- `debate__<timestamp>.txt` — Human-readable debate transcript
- `debate__<timestamp>.json` — Structured data with all questions and answers
