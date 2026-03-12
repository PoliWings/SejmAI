# SejmAI

Research project investigating political bias in Polish Large Language Models through the concept of **agonistic pluralism** — steering ideological tendencies by fine-tuning on real parliamentary speeches.

The project constructs the **Polish Parliamentary Speeches Corpus (PPSC)**, fine-tunes ideologically distinct LoRA adapters (left-wing and right-wing) on top of [Bielik-11B-v2.2-Instruct](https://huggingface.co/speakleash/Bielik-11B-v2.2-Instruct), and evaluates their political alignment using a custom benchmark.

> **Paper:** _"Asymmetric Ideological LLM Fine-Tuning in the Polish Political Domain"_
>
> **Authors:** Damian Jankowski, Radosław Gajewski, Jan Barczewski, Maciej Sikora, Jan Majkutewicz (Gdańsk University of Technology)

## Hugging Face Resources

| Resource         | URL                                                                                                                              |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Left-wing model  | [PoliWings/opposing-views-left-wing](https://huggingface.co/PoliWings/opposing-views-left-wing)                                  |
| Right-wing model | [PoliWings/opposing-views-right-wing](https://huggingface.co/PoliWings/opposing-views-right-wing)                                |
| Dataset          | [PoliWings/Polish-Parliamentary-Speeches-Corpus](https://huggingface.co/datasets/PoliWings/Polish-Parliamentary-Speeches-Corpus) |

## Project Structure

```
SejmAI/
├── scraper/              # Data collection from the Polish Sejm API
├── data_processor/       # Speech classification and dataset formatting (SFT/DPO)
├── data_visualization/   # Training metrics and evaluation result plots
├── fine_tuning/          # LoRA/QLoRA fine-tuning scripts (local & remote)
├── test_survey/          # Political alignment benchmark (267 statements, 5 domains)
├── debate_simulation/    # Simulated political debates between left/right models
└── file_manager.sh       # Utility for managing files on a remote hosting service
```

## Pipeline Overview

The end-to-end workflow follows these stages:

1. **Scrape** — Collect parliamentary speeches and MP metadata from [api.sejm.gov.pl](https://api.sejm.gov.pl).
2. **Process** — Classify MPs as left/right, clean transcripts, and format into SFT and DPO datasets.
3. **Fine-tune** — Train LoRA adapters on left-wing and right-wing speech subsets.
4. **Evaluate** — Measure political bias using a 267-question benchmark across 5 policy domains.
5. **Simulate** — Run debates between the two fine-tuned models on political topics.

## Key Results

| Model                                 | Bias Score |
| ------------------------------------- | ---------- |
| Base model (Bielik-11B-v2.2-Instruct) | 72.18%     |
| Left-wing fine-tuned                  | 82.85%     |
| Right-wing fine-tuned                 | 54.18%     |

The fine-tuning exhibits **asymmetric behavior** — left-wing training amplifies the base model's existing left bias, while right-wing training primarily neutralizes it rather than pushing the model to the right.

## Quick Start

Each module has its own README with detailed setup instructions. The general requirements are:

- Python 3.10+
- CUDA-compatible GPU (for fine-tuning and local inference)
- [Conda](https://docs.conda.io/) (recommended for environment management)

```bash
# Clone the repository
git clone https://github.com/PoliWings/SejmAI.git
cd SejmAI
```

From here, navigate to the desired module (e.g., `data_processor/`, `fine_tuning/`, etc.) and follow the specific README instructions for setup and usage.

## Acknowledgements

Supported by the _"Cloud Artificial Intelligence Service Engineering (CAISE) platform"_ project (No. KPOD.05.10-IW.10-0005/24) as part of the European IPCEI-CIS program, financed by NRRP funds.

## License

[Apache 2.0](LICENSE)
