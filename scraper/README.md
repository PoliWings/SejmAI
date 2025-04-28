# Sejm API Scraper

## Description
This folder contains two Python scripts for downloading and processing data from the Polish Sejm API. The scripts fetch speech transcripts from Sejm sessions and data about Members of Parliament (MPs) and parliamentary clubs. The data is saved as JSON files for further analysis.

## Scripts Overview

### `speeches.py`
This script downloads speeches of MPs from a specific term of the Polish Sejm. It retrieves the session proceedings and individual MP speeches, saving them in JSON files. 

#### Features:
- Retrieves proceedings of Sejm sessions for a given term.
- Downloads MP speech transcripts for selected dates.
- Saves speeches in JSON format.
- Handles errors and supports interruption and resumption of the process.

### `mp_clubs.py`
This script fetches data about MPs and parliamentary clubs from a specific term of the Polish Sejm. It retrieves a list of MPs (including those who no longer serve) and parliamentary clubs, saving them as JSON files.

#### Features:
- Retrieves a list of all MPs for a given term.
- Retrieves a list of parliamentary clubs and groups.
- Saves MPs and clubs data in JSON format.
- Handles errors and supports interruption and resumption of the process.

## Installation

To run either script, you need Python 3 installed along with the `requests` and `beautifulsoup4` libraries (for `speeches.py`). You can install the required libraries by running:

```
pip install -r requirements.txt
```

## Running the Scripts

### `speeches.py`
To execute the script, use the following command in your terminal, providing the term number for which you want to download the data:

```
python speeches.py --term <term_number> [--force]
```

#### Arguments:
- `--term <term_number>`: The Sejm term number (e.g., `10`).
- `--force`: Optional flag that deletes previously saved files in the `output/` folder.

### `mp_clubs.py`
To execute this script, use the following command:

```
python mp_clubs.py --term <term_number> [--force]
```

#### Arguments:
- `--term <term_number>`: The Sejm term number (e.g., `10`).
- `--force`: Optional flag that deletes previously saved files in the `output/` folder.

## Folder Structure

Both scripts save their output in the `output/` directory, organized by term numbers. The resulting files include JSON data for the speeches, MPs, and clubs.

### `speeches.py` Output:
The speeches are saved in the following folder structure:
```
output/
    ├── term<term_number>/
    │   ├── <session_number>/
    │   │   ├── <date>/
    │   │   │   ├── <statement_number>.json
    │   │   │   └── agenda.json
    │   └── ...
```

### `mp_clubs.py` Output:
The MPs and clubs data are saved in the following structure:
```
output/
    ├── mp_clubs/
    │   ├── term<term_number>/
    │   │   ├── clubs.json
    │   │   └── members.json
    └── ...
```