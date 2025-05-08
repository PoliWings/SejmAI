import os
import requests
from dotenv import load_dotenv
import urllib3
import time

MAX_RETRIES = 5
system_prompt = "Odpowiadaj krótko, precyzyjnie i wyłącznie w języku polskim."

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def prompt_model(prompt):
    load_dotenv()
    assert "LLM_USERNAME" in os.environ, f"Environment variable LLM_USERNAME must be set"
    assert "LLM_PASSWORD" in os.environ, f"Environment variable LLM_PASSWORD must be set"
    assert "LLM_URL" in os.environ, f"Environment variable LLM_URL must be set"

    url = os.getenv("LLM_URL")
    auth = (os.getenv("LLM_USERNAME"), os.getenv("LLM_PASSWORD"))
    auth_kwargs = {"auth": auth, "verify": False}

    data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_length": 64,
        "temperature": 0.7,
    }
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.put(
                url=url,
                json=data,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                **auth_kwargs,
            )
            response.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e} retry: {retries}")
            retries += 1
            if retries >= MAX_RETRIES:
                raise e
            time.sleep(retries)

    response_json = response.json()
    return response_json["response"]
