import argparse
import os
import shutil
import requests
from bs4 import BeautifulSoup
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://api.sejm.gov.pl/sejm"
stop_event = threading.Event()


def get_proceedings(term):
    if stop_event.is_set():
        return []

    url = f"{BASE_URL}/term{term}/proceedings"
    response = requests.get(url, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


def get_transcripts(term, session_num, date):
    if stop_event.is_set():
        return {}

    url = f"{BASE_URL}/term{term}/proceedings/{session_num}/{date}/transcripts"
    response = requests.get(url, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


def get_speech_html(term, session_num, date, statement_num):
    if stop_event.is_set():
        return "", ""

    url = f"{BASE_URL}/term{term}/proceedings/{session_num}/{date}/transcripts/{statement_num}"
    response = requests.get(url, headers={"Accept": "text/html"})
    response.raise_for_status()
    return response.text, url


def extract_title_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    title_tag = soup.find('title')
    return title_tag.text.strip() if title_tag else ""


def remove_html_tags(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()


def extract_context_and_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    context_tag = soup.find('p', class_='punkt-tytul')
    context = ' '.join(context_tag.get_text().split()) if context_tag else ""
    text = [' '.join(p.get_text().split())
            for p in soup.find_all('p') if not p.get('class')]
    return context, " ".join(text)


def save_speech(term, session_num, date, statement_num, speech_data):
    if stop_event.is_set():
        return

    directory = os.path.join("output/speeches", f"term{term}", str(session_num), date)
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"{statement_num}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(speech_data, f, ensure_ascii=False, indent=2)
    print(f"Saved speech: {filename}")


def save_proceeding(term, session_num, session_title, agenda_html):
    if stop_event.is_set():
        return

    soup = BeautifulSoup(agenda_html, "html.parser")
    agenda_items = [" ".join(li.stripped_strings)
                    for li in soup.find_all("li")]

    session_url = f"{BASE_URL}/term{term}/proceedings/{session_num}"

    directory = os.path.join("output/speeches", f"term{term}", str(session_num))
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, "agenda.json")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "title": session_title,
            "agenda": agenda_items,
            "link": session_url
        }, f, ensure_ascii=False, indent=2)

    print(f"Saved proceeding: {filename}")


def process_statement(term, session_num, date, statement):
    if stop_event.is_set():
        return

    statement_num = statement.get("num")
    speaker = statement.get("name")

    try:
        html_text, link = get_speech_html(
            term, session_num, date, statement_num)
        if stop_event.is_set():
            return

        title = extract_title_from_html(html_text)
        context, text = extract_context_and_text(html_text)
    except Exception as e:
        print(
            f"Error retrieving statement {statement_num} in session {session_num} on {date}: {e}")
        return

    speech_data = {
        "title": title,
        "speaker": speaker,
        "context": context,
        "text": text,
        "link": link
    }

    save_speech(term, session_num, date, statement_num, speech_data)


def process_date(term, session_num, date):
    if stop_event.is_set():
        return

    try:
        transcripts_data = get_transcripts(term, session_num, date)
        if stop_event.is_set():
            return
    except Exception as e:
        print(
            f"Error retrieving statements for session {session_num} on {date}: {e}")
        return

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_statement, term, session_num, date, statement)
                   for statement in transcripts_data.get("statements", [])]

        try:
            for future in as_completed(futures):
                if stop_event.is_set():
                    for f in futures:
                        f.cancel()
                    break
                future.result()
        except KeyboardInterrupt:
            print("Stopping processing...")
            stop_event.set()
            for future in futures:
                future.cancel()


def process_term(term):
    if stop_event.is_set():
        return

    proceedings = get_proceedings(term)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []

        try:
            for session in proceedings:
                if stop_event.is_set():
                    break

                session_num = session.get("number")
                if not session_num:
                    continue

                agenda = session.get("agenda", "").strip()
                session_title = session.get("title", "").strip()

                save_proceeding(term, session_num, session_title, agenda)

                dates = session.get("dates", [])
                for date in dates:
                    if stop_event.is_set():
                        break
                    futures.append(executor.submit(
                        process_date, term, session_num, date))

            for future in as_completed(futures):
                if stop_event.is_set():
                    for f in futures:
                        f.cancel()
                    break
                future.result()
        except KeyboardInterrupt:
            print("Stopping processing...")
            stop_event.set()
            for future in futures:
                future.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetches politicians' speeches from a given term of the Polish Sejm.")
    parser.add_argument("--term", type=int, required=True,
                        help="Term number (e.g., 10)")
    parser.add_argument("--force", action="store_true",
                        help="Deletes previous files")
    args = parser.parse_args()

    if args.force:
        shutil.rmtree("output/speeches", ignore_errors=True)

    try:
        process_term(args.term)
    except KeyboardInterrupt:
        print("Process interrupted. Exiting...")
        stop_event.set()
