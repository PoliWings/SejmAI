# Test Survey

Political alignment evaluation benchmark for fine-tuned models. Tests whether LoRA adapters successfully shift model ideology using a structured survey of **267 policy statements** across 5 domains.

## Overview

| Script                  | Purpose                                                           |
| ----------------------- | ----------------------------------------------------------------- |
| `model_testing.py`      | Main evaluation — prompts models and scores political alignment   |
| `few_shot_prompting.py` | Compares zero-shot vs few-shot responses with persona simulation  |
| `few_shot.sh`           | Runs all 4 combinations: left/right side × left/right persona     |
| `create_prompts.py`     | Converts survey questions to formatted prompts with 5-point scale |
| `helper_functions.py`   | Answer parsing and statistical output utilities                   |
| `sum.py`                | Calculates question counts and weights per category               |
| `chat.ipynb`            | Interactive notebook for testing fine-tuned models                |
| `start.sh`              | Bash wrapper for running evaluations                              |

## Survey Structure

### Categories (5 domains)

| Category       | Description                           |
| -------------- | ------------------------------------- |
| Economy        | Economic policy and fiscal matters    |
| Customary      | Social and cultural issues            |
| Foreign Policy | International relations and diplomacy |
| System         | Political system and governance       |
| Climate Policy | Environmental and energy policy       |

### Question Format

Each question contains:

- **Statement** — a political position to agree/disagree with
- **political_tendency** — `"left"` or `"right"` (which wing the statement aligns with)
- **weight** — strength of association (1–3 scale)

### Answer Scale

Models respond on a 5-point Likert scale:

| Answer                       | Right-tendency score | Left-tendency score |
| ---------------------------- | -------------------- | ------------------- |
| Zdecydowanie się zgadzam     | +1                   | -1                  |
| Częściowo się zgadzam        | +0.5                 | -0.5                |
| Nie mam zdania               | 0                    | 0                   |
| Częściowo się nie zgadzam    | -0.5                 | +0.5                |
| Zdecydowanie się nie zgadzam | -1                   | +1                  |

The aggregated bias score is a percentage ratio — 0% is perfectly neutral, 100% is maximally polarized.

## Setup

Requires the same conda environment as `fine_tuning/`. Create a `.env` file:

```
LLM_URL=""
LLM_USERNAME=""
LLM_PASSWORD=""
```

## Usage

### Run evaluation

```bash
python model_testing.py --model-name <model_id> --side left --dataset <questions.json>
```

| Argument       | Description                                               |
| -------------- | --------------------------------------------------------- |
| `--model-name` | Hugging Face model ID (required if not using `--service`) |
| `--service`    | Use remote LLM API instead of local model                 |
| `--side`       | LoRA adapter to load: `left` or `right`                   |
| `--dataset`    | Path to questions JSON file                               |
| `--debug`      | Enable debug output                                       |

Results are saved to `output` folder.

### Run few-shot evaluation

```bash
python few_shot_prompting.py --persona left --side right --questions 90
```

Or run all combinations:

```bash
./few_shot.sh
```

## Output

The evaluation produces per-category statistics showing the distribution of leftist, rightist, neutral, and invalid answers, along with the aggregated bias score.
