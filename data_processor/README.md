# Sejm API Data processor

## Description
This folder contains two Python scripts: one for creating MPs classification as left-wing or right-wing and the other for formatting the data into training sets for the LLMs. The scripts require data scrapped by scripts located in scrapper folder.

## Scripts Overview

### `map_members.py`
This script creates `member_mapping.json` file with MPs classification as left or right wing.

### `process_data.py`
This script generates datasets for SFT and DPO model training, based on scrapped speeches. For each training method two datasets are created: `left.json` and `right.json`.

### `llm_connection.py`
Used by `process_data.py` to prompt LLM to generate contexts. It's not meant to be run directly by user.

## Installation

To run scripts, you need Python 3 installed along with required libraries (specified in `requirements.txt`). You can install the required libraries by running:

```
pip install -r requirements.txt
```
It's recommended to use some environment such as venv or conda.

## Running the Scripts

### `map_members.py`
1. Get all the clubs from scrapped data by running the script with `--get_clubs` argument:
```
python map_members.py --get_clubs
```
2. The script will generate a `club_mapping.json` file. Enter it and manually fill the mapping as "left", "right" or "".
3. Run the script with `--get_members` argument to map individual members based on `club_mapping.json`:
```
python map_members.py --get_members
```
4. The script will generate `member_mapping.json` file with MPs classification.

### `process_data.py`
1. Have `member_mapping.json` from `map_members.py`.
2. Run the script to generate datasets:
```
python process_data.py
```
3. The datasets will be saved in the `./output` folder in `JSON` format using chat notation.
4. To get better results when training with this dataset it's recommended to generate contexts, as a lot of the scrapped speeches don't have one. To do this:

    - Create a `.env` file with the following variables:
    ```
    LLM_URL=""
    LLM_USERNAME=""
    LLM_PASSWORD="" 
    ```
    - Run the `process_data.py` script again but this time with `--gen_context` flag:
    ```
    python process_data.py --gen_context
    ```
    This process can take up to a few hours depending on the speed of your LLM, number of scrapped speeches or percentage of already present contexts in the data. Because of that the script will save a checkpoint every 500 speeches. It will be used when rerunning the script in case it fails during the process.