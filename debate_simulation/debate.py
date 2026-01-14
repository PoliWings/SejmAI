import requests
import json
import urllib3
import argparse

from tqdm import tqdm

import os
from datetime import datetime
import random
from dotenv import load_dotenv

load_dotenv("../.env")

import warnings

warnings.filterwarnings("ignore", message="Found missing adapter keys while loading the checkpoint:")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEVICE = "cuda"
SIDES = ["left", "right"]
QUESTION_CATEGORIES = [
    "Polityka zagraniczna",
    "Polityka wewnętrzna",
    "Gospodarka",
    "Edukacja",
    "Ochrona zdrowia",
    "Bezpieczeństwo narodowe",
    "Polityka społeczna",
    "Prawa obywatelskie",
    "Środowisko i zmiany klimatyczne",
    "Energetyka",
    "Rolnictwo i obszary wiejskie",
    "Transport i infrastruktura",
    "Technologia i cyfryzacja",
    "Praworządność i wymiar sprawiedliwości",
    "Imigracja i integracja",
    "Kultura i dziedzictwo narodowe",
    "Polityka mieszkaniowa",
    "Podatki i finanse publiczne",
    "Rynek pracy i zatrudnienie",
    "Unia Europejska i integracja europejska",
]


system_prompt = (
    "Jesteś uczestnikiem telewizyjnej debaty politycznej. Twoim zadaniem jest odpowiadać na pytania w sposób przekonujący, logiczny i emocjonalnie wyważony. "
    "Odpowiadaj wyłącznie w języku polskim, z zachowaniem naturalnego tonu rozmowy — jak w studiu telewizyjnym, a nie w parlamencie. "
    "Pod żadnym pozorem nie używaj zwrotów typowych dla wystąpień sejmowych ani formalnych przemówień, takich jak: "
    "'Szanowni Państwo Posłowie', 'Panie Marszałku', 'Wysoka Izbo', 'Panie i Panowie Posłowie', 'Wysoka Komisjo' i podobnych. "
    "Jeśli masz wątpliwość, jak się zwrócić — zwróć się po prostu do słuchaczy lub do swojego przeciwnika, używając neutralnych form, np. 'Państwo', 'drodzy widzowie', 'mój rozmówco'. "
    "Twoim celem jest przekonać odbiorców do swojego stanowiska poprzez rzeczowe argumenty, dane i przykłady. "
    "Nie przemawiaj jak polityk w Sejmie — mów jak uczestnik publicznej debaty. "
    "Twoja wypowiedź powinna być krótka, spójna i zakończona podsumowaniem stanowiska. "
    "Nie powtarzaj tych samych zdań ani słów. Zakończ odpowiedź, gdy przedstawisz swoje stanowisko."
)

questions = []

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, help="Name of the local model")
parser.add_argument("--service", action="store_true", help="Run tests via service API")
parser.add_argument("--questions", type=str, help="Path to the JSON file with questions")
parser.add_argument("--ask-questions", type=int, help="Number of questions each model will ask the other model")

args = parser.parse_args()

if not args.service and not args.model_name:
    parser.error("--model-name must be specified if --service is not used")

use_service = args.service
model_name = args.model_name
if model_name:
    base_name = model_name.replace("/", "_")

if not args.questions and not args.ask_questions:
    parser.error("--questions or --ask-questions must be specified")

filename = args.questions
num_questions = args.ask_questions

local_model = None
local_tokenizer = None

assert "LLM_USERNAME" in os.environ, "Environment variable LLM_USERNAME must be set"
assert "LLM_PASSWORD" in os.environ, "Environment variable LLM_PASSWORD must be set"
auth = (os.getenv("LLM_USERNAME"), os.getenv("LLM_PASSWORD"))
auth_kwargs = {"auth": auth, "verify": False}


def send_chat_prompt(prompt, side):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    data = {
        "messages": messages,
        "max_length": 1024,
        "temperature": 0.7,
    }
    data["lora_adapter"] = f"opposing_views__{side}_lora_module"
    try:
        response = requests.put(
            f"{os.getenv('LLM_URL')}/llm/prompt/chat",
            json=data,
            timeout=120,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            **auth_kwargs,
        )
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return None
    response.raise_for_status()
    response = response.json().get("response")
    return response


def prepare_answer_prompt(question, asker):
    prompt = f"Odpowiedz na zadane pytanie zgodnie ze swoimi poglądami. Pytanie zadaje {asker}. Pytanie: {question['question']}"
    if len(question["answers"]) > 0:
        prompt += f"\nTwój przeciwnik na to samo pytanie odpowiedział: {question['answers'][0]['answer']}"
    else:
        prompt += "\nTwój przeciwnik jeszcze nie odpowiedział na to pytanie."
    return prompt


def prepare_gen_question_prompt(topic):
    prompt = f"Jako uczestnik debaty politycznej, przygotuj pytanie do twojego przeciwnika dotyczące tematu: {topic}. Pytanie powinno być otwarte i skłaniać do dyskusji."
    return prompt


if __name__ == "__main__":
    if filename:
        with open(filename, "r", encoding="utf-8") as source:
            data = json.load(source)
    else:
        data = []

    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"output/debate__{timestamp}.txt", "w", encoding="utf-8") as dest:
        dest.write(f"PYTANIA OD PROWADZĄCEGO DEBATĘ\n\n\n")
        for question in tqdm(data, desc="Asking questions from file"):
            dest.write(f"Question: {question['question']}\n\n")
            dest.flush()
            question["answers"] = []
            question["asked_by"] = "mediator"
            random_sides = SIDES[:]
            random.shuffle(random_sides)
            for side in random_sides:
                answer = send_chat_prompt(prepare_answer_prompt(question, "prowadzący debatę"), side)
                answer = {"side": side, "answer": answer}
                question["answers"].append(answer)
                dest.write(f"Side: {answer['side']}\nAnswer: {answer['answer']}\n\n")
                dest.flush()
            questions.append(question)

        dest.write(f"\n\nPYTANIA GENEROWANE PRZEZ MODELE\n\n\n")
        if num_questions:
            for _ in tqdm(range(num_questions), desc="Generating and asking new questions"):
                random_sides = SIDES[:]
                random.shuffle(random_sides)
                for side in random_sides:
                    topic = random.choice(QUESTION_CATEGORIES)
                    question = send_chat_prompt(prepare_gen_question_prompt(topic), side)
                    question = {"question": question, "answers": [], "asked_by": side}
                    dest.write(f"Question generated by {side}: {question['question']}\n\n")
                    dest.flush()
                    for other_side in random_sides:
                        if other_side == side:
                            continue
                        answer = send_chat_prompt(prepare_answer_prompt(question, "twój przeciwnik"), other_side)
                        answer = {"side": other_side, "answer": answer}
                        question["answers"].append(answer)
                        dest.write(f"Side: {answer['side']}\nAnswer: {answer['answer']}\n\n")
                        dest.flush()
                    questions.append(question)
    with open(f"output/debate__{timestamp}.json", "w", encoding="utf-8") as dest:
        json.dump(questions, dest, ensure_ascii=False, indent=4)
