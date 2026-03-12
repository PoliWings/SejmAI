# Data Visualization

Scripts for generating plots and charts from training metrics and evaluation results.

## Overview

| Script                                | Purpose                                                                             |
| ------------------------------------- | ----------------------------------------------------------------------------------- |
| `data_visualization.py`               | Compares training metrics (loss, learning rate, etc.) between left and right models |
| `bar_chart.py`                        | Generates bar charts for human evaluation (Turing test) results                     |
| `average_scores_for_categories.py`    | Calculates and displays average political bias scores per category                  |
| `data_length_analysis.py`             | Analyzes speech length distributions in the dataset                                 |
| `graphs_grouped_by_categories.py`     | Generates evaluation plots grouped by question categories                           |
| `graphs_grouped_by_system_prompts.py` | Generates evaluation plots grouped by system prompt variants                        |
| `txt_to_json_summaries.py`            | Converts text evaluation outputs into structured JSON for plotting                  |

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:** `matplotlib`, `numpy`

## Input Data

The following statistics files are required for generating the paper figures.

| File                            | Used by                                                               |
| ------------------------------- | --------------------------------------------------------------------- |
| `categories_statistics.json`    | `graphs_grouped_by_categories.py`, `average_scores_for_categories.py` |
| `system_prompt_statistics.json` | `graphs_grouped_by_system_prompts.py`                                 |

## Usage

### Paper Figures

| Figure                                         | Script                                | Output                                 |
| ---------------------------------------------- | ------------------------------------- | -------------------------------------- |
| Fig. 1 — Bias scores grouped by categories     | `graphs_grouped_by_categories.py`     | `plots/averages/general.png`           |
| Fig. 2 — Bias scores grouped by system prompts | `graphs_grouped_by_system_prompts.py` | `plots/system_prompt_all.png`          |
| Fig. 3 — Turing test results                   | `bar_chart.py`                        | `plots/percentage_correct_colored.png` |

```bash
python graphs_grouped_by_categories.py
python graphs_grouped_by_system_prompts.py
python bar_chart.py
```

### Training metric comparison

Requires `trainer_state_left.json` and `trainer_state_right.json`:

```bash
python data_visualization.py
```

### Other utilities

```bash
python average_scores_for_categories.py   # Print average bias scores per category
python data_length_analysis.py            # Analyze speech length distributions
python txt_to_json_summaries.py           # Convert text outputs to JSON for plotting
```

## Output

All generated plots are saved to the `plots/` directory.
