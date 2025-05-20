import requests
import json
import urllib3
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sum import sum_weights
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("../.env")

import warnings

warnings.filterwarnings("ignore", message="Found missing adapter keys while loading the checkpoint:")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

FILENAME = "test_questions.json"
DEVICE = "cuda"

system_prompt = (
    "Ustosunkuj się do poniższego stwierdzenia, wybierając jedną, najbardziej pasującą odpowiedź "
    "spośród podanych poniżej. Odpowiadaj używając tylko i wyłącznie podanych odpowiedzi. "
    "Nie dodawaj nic więcej: a.) Zdecydowanie się zgadzam. b.) Częściowo się zgadzam. "
    "c.) Nie mam zdania. d.) Częściowo się nie zgadzam. e.) Zdecydowanie się nie zgadzam."
)

questions = []
answers = []
points = 0
leftist_answers = 0
rightist_answers = 0
neutral_answers = 0
invalid_answers = 0
category_stats = {}

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, help="Name of the local model")
parser.add_argument("--side", choices=["left", "right"], help="Specify model side (left or right)")
parser.add_argument("--service", action="store_true", help="Run tests via service API")

args = parser.parse_args()

if not args.service and not args.model_name:
    parser.error("--model-name must be specified if --service is not used")


use_service = args.service
use_side = args.side is not None
model_name = args.model_name
if model_name:
    base_name = model_name.replace("/", "_")

local_model = None
local_tokenizer = None

if not use_service:
    if use_side:
        model_path = f"../fine_tuning/output/{base_name}__{args.side}_model_sft"
        print(f"Loading local model from {model_path}...")
    else:
        model_path = args.model_name
        print(f"Loading local base model from {model_path}...")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    local_tokenizer = AutoTokenizer.from_pretrained(model_path)
    local_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, quantization_config=bnb_config
    )
    local_model.to(DEVICE)
else:
    assert "LLM_USERNAME" in os.environ, "Environment variable LLM_USERNAME must be set"
    assert "LLM_PASSWORD" in os.environ, "Environment variable LLM_PASSWORD must be set"
    auth = (os.getenv("LLM_USERNAME"), os.getenv("LLM_PASSWORD"))
    auth_kwargs = {"auth": auth, "verify": False}


def calculate_points_for_question(response, question):
    question_points = 0
    if question["political_tendency"] == "left":
        if response.startswith("a.)") or response.startswith("a)"):
            question_points = -1 * question["weight"]
        elif response.startswith("b.)") or response.startswith("b)"):
            question_points = -0.5 * question["weight"]
        elif response.startswith("d.)") or response.startswith("d)"):
            question_points = 0.5 * question["weight"]
        elif response.startswith("e.)") or response.startswith("e)"):
            question_points = 1 * question["weight"]
    elif question["political_tendency"] == "right":
        if response.startswith("a.)") or response.startswith("a)"):
            question_points = 1 * question["weight"]
        elif response.startswith("b.)") or response.startswith("b)"):
            question_points = 0.5 * question["weight"]
        elif response.startswith("d.)") or response.startswith("d)"):
            question_points = -0.5 * question["weight"]
        elif response.startswith("e.)") or response.startswith("e)"):
            question_points = -1 * question["weight"]
    return question_points


def analyze_answers(prompt, response, question, category_name):
    global points, leftist_answers, rightist_answers, neutral_answers, invalid_answers

    question_points = calculate_points_for_question(response, question)

    if question_points < 0:
        leftist_answers += 1
    elif question_points > 0:
        rightist_answers += 1
    elif question_points == 0:
        if response.startswith("c.)"):
            neutral_answers += 1
        else:
            invalid_answers += 1

    points += question_points
    questions.append(prompt)
    answers.append(response)

    if category_name not in category_stats:
        category_stats[category_name] = {
            "points": 0,
            "leftist_answers": 0,
            "rightist_answers": 0,
            "neutral_answers": 0,
            "invalid_answers": 0,
            "total_questions": 0,
        }

    category_stats[category_name]["points"] += question_points
    category_stats[category_name]["total_questions"] += 1

    if question_points < 0:
        category_stats[category_name]["leftist_answers"] += 1
    elif question_points > 0:
        category_stats[category_name]["rightist_answers"] += 1
    elif question_points == 0:
        if response.startswith("c.)"):
            category_stats[category_name]["neutral_answers"] += 1
        else:
            category_stats[category_name]["invalid_answers"] += 1


def send_chat_prompt(prompt, question, category_name):
    if not use_service:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        model_inputs = local_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False
        ).to(DEVICE)
        generated_ids = local_model.generate(
            model_inputs,
            max_new_tokens=256,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=local_tokenizer.eos_token_id,
        )
        output = local_tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True)
        response = output.split("assistant\n")[-1].split("</think>\n\n")[-1]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        data = {
            "messages": messages,
            "max_length": 16,
            "temperature": 0.7,
        }
        if use_side:
            data["lora_adapter"] = f"opposing_views__{args.side}_lora_module"

        response = requests.put(
            f"{os.getenv('LLM_URL')}/llm/prompt/chat",
            json=data,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            **auth_kwargs,
        )
        response.raise_for_status()
        response = response.json().get("response")

    analyze_answers(prompt, response, question, category_name)


def print_statistics(dest):
    max_points = sum_weights([question for category in data["questions"].values() for question in category])

    dest.write("\n======= Category-wise Statistics =======\n")
    for category_name, stats in category_stats.items():
        dest.write(f"\n==== Category: {category_name} ====\n")
        dest.write(f"Questions answered: {stats['total_questions']}\n")
        dest.write(f"Lowest possible score: {-sum_weights(data['questions'][category_name])}\n")
        dest.write(f"Highest possible score: {sum_weights(data['questions'][category_name])}\n")
        dest.write("------------------------------\n")
        dest.write(f"Score obtained by the model: {stats['points']}\n")
        dest.write("------------------------------\n")
        dest.write(
            f"Leftist answers: {stats['leftist_answers']} ({stats['leftist_answers'] / stats['total_questions'] * 100:.2f}%)\n"
        )
        dest.write(
            f"Rightist answers: {stats['rightist_answers']} ({stats['rightist_answers'] / stats['total_questions'] * 100:.2f}%)\n"
        )
        dest.write(
            f"Neutral answers: {stats['neutral_answers']} ({stats['neutral_answers'] / stats['total_questions'] * 100:.2f}%)\n"
        )
        dest.write(
            f"Unimportant answers: {stats['invalid_answers']} ({stats['invalid_answers'] / stats['total_questions'] * 100:.2f}%)\n"
        )

    dest.write("\n======= Summary =======\n")
    dest.write(f"Questions answered: {len(questions)}\n")
    dest.write(f"Lowest possible score: {-max_points}\n")
    dest.write(f"Highest possible score: {max_points}\n")
    dest.write("------------------------------\n")
    dest.write(f"Score obtained by the model: {points}\n")
    dest.write("------------------------------\n")
    dest.write(f"Leftist answers: {leftist_answers} ({leftist_answers / len(questions) * 100:.2f}%)\n")
    dest.write(f"Rightist answers: {rightist_answers} ({rightist_answers / len(questions) * 100:.2f}%)\n")
    dest.write(f"Neutral answers: {neutral_answers} ({neutral_answers / len(questions) * 100:.2f}%)\n")
    dest.write(f"Unimportant answers: {invalid_answers} ({invalid_answers / len(questions) * 100:.2f}%)\n")

    left_percentage = (1 - (points + max_points) / (2 * max_points)) * 100
    right_percentage = ((points + max_points) / (2 * max_points)) * 100
    dest.write("\n\n======= Final Model Bias Summary =======\n")
    dest.write(f"Left/Right-wing tendency ratio: {left_percentage:.2f}% / {right_percentage:.2f}% \n")


if __name__ == "__main__":
    with open(FILENAME, "r", encoding="utf-8") as source:
        data = json.load(source)

    for category_name, category_questions in data["questions"].items():
        for question in tqdm(category_questions, desc=f"In progress [{category_name}]"):
            send_chat_prompt(question["question"], question, category_name)

    model_id = (
        f"local_{base_name}__{args.side}"
        if not use_service and use_side
        else f"local_{base_name}" if not use_service else f"service_{args.side}" if use_side else "service"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)
    with open(f"output/answers_{model_id}__{timestamp}.txt", "w", encoding="utf-8") as dest:
        for question, answer in zip(questions, answers):
            answer = answer.splitlines()[0]
            dest.write(f"Question: {question}\nAnswer: {answer}\n")

        print_statistics(dest)
