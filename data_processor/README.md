# Sejm API Data processor

## Description
This folder contains two Python scripts: one for creating MPs classification as left-wing or right-wing and the other for formatting the data into training sets for the LLMs. The scripts require data scrapped by scripts located in scrapper folder.

## Scripts Overview

### `map_members.py`
This script creates `member_mapping.json` file with MPs classification as left or right wing.

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