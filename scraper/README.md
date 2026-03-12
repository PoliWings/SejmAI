# Scraper

Data collection module for downloading parliamentary speeches and MP metadata from the [Polish Sejm API](https://api.sejm.gov.pl).

## Overview

| Script        | Purpose                                                       |
| ------------- | ------------------------------------------------------------- |
| `speeches.py` | Downloads MP speech transcripts from Sejm session proceedings |
| `mp_clubs.py` | Fetches lists of MPs and parliamentary club affiliations      |

Both scripts support **resumable downloads** — interrupted runs can be continued without re-downloading existing data.

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:** `requests`, `beautifulsoup4`

## Usage

### Scraping speeches

```bash
python speeches.py --term <term_number> [--force]
```

| Argument  | Description                                    |
| --------- | ---------------------------------------------- |
| `--term`  | Sejm term number (e.g., `10`)                  |
| `--force` | Delete previously saved output and re-download |

Each speech is saved as a JSON file containing the title, speaker name, context, text, and link to the original transcript.

### Scraping MP & club data

```bash
python mp_clubs.py --term <term_number> [--force]
```

Retrieves all MPs (including former ones) and parliamentary clubs for the specified term.

## Output Structure

```
output/
├── speeches/
│   └── term<N>/
│       └── session<N>/
│           └── <date>/
│               ├── agenda.json
│               └── <statement_number>.json
└── mp_clubs/
    └── term<N>/
        ├── clubs.json
        └── members.json
```

## Data Source

All data is sourced from the public [Sejm API](https://api.sejm.gov.pl/sejm). Each parliamentary term has its own license — see the Sejm website for details (e.g., [10th term license](https://www.sejm.gov.pl/sejm10.nsf/page.xsp/copyright)).
