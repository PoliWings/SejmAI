import argparse
import os
import shutil
import requests
import json
import threading

BASE_URL = "https://api.sejm.gov.pl/sejm"
stop_event = threading.Event()


def get_members(term):
    if stop_event.is_set():
        return []

    url = f"{BASE_URL}/term{term}/MP"
    response = requests.get(url, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


def get_clubs(term):
    if stop_event.is_set():
        return []

    url = f"{BASE_URL}/term{term}/clubs"
    response = requests.get(url, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


def save_members(term, members_data):
    if stop_event.is_set():
        return

    directory = os.path.join("output/mp_clubs", f"term{term}")
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, "members.json")

    for member in members_data:
        member["link"] = f"{BASE_URL}/term{term}/MP/{member.get('id')}"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(members_data, f, ensure_ascii=False, indent=2)

    print(f"Saved members: {filename}")


def save_clubs(term, clubs_data):
    if stop_event.is_set():
        return

    directory = os.path.join("output/mp_clubs", f"term{term}")
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, "clubs.json")

    for club in clubs_data:
        club["link"] = f"{BASE_URL}/term{term}/clubs/{club.get('id')}"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(clubs_data, f, ensure_ascii=False, indent=2)

    print(f"Saved clubs: {filename}")


def process_term(term):
    if stop_event.is_set():
        return

    try:
        members_data = get_members(term)
        clubs_data = get_clubs(term)

        save_members(term, members_data)
        save_clubs(term, clubs_data)

    except Exception as e:
        print(f"Error retrieving data for term {term}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetches MPs and clubs data from a given term of the Polish Sejm.")
    parser.add_argument("--term", type=int, required=True, help="Term number (e.g., 10)")
    parser.add_argument("--force", action="store_true", help="Deletes previous files")
    args = parser.parse_args()

    if args.force:
        shutil.rmtree("output/mp_clubs", ignore_errors=True)

    try:
        process_term(args.term)
    except KeyboardInterrupt:
        print("Process interrupted. Exiting...")
        stop_event.set()
