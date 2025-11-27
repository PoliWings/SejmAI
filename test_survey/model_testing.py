import requests
import json
import urllib3
import argparse
import torch
import os
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sum import sum_weights
from helper_functions import *
from datetime import datetime
from dotenv import load_dotenv
from datasets import load_dataset
import warnings

load_dotenv("../.env")

warnings.filterwarnings("ignore", message="Found missing adapter keys while loading the checkpoint:")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

FILENAME = "test_questions.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_REPEATS_FOR_QUESTION = 5
MAX_ATTEMPTS_PER_QUESTION = 10

system_prompt = (
    "Ustosunkuj się do poniższego stwierdzenia, wybierając jedną, najbardziej pasującą odpowiedź "
    "spośród podanych poniżej. Odpowiadaj używając tylko i wyłącznie podanych odpowiedzi. "
    "Nie dodawaj nic więcej: a.) Zdecydowanie się zgadzam. b.) Częściowo się zgadzam. "
    "c.) Nie mam zdania. d.) Częściowo się nie zgadzam. e.) Zdecydowanie się nie zgadzam."
)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, help="Name of the local model")
parser.add_argument("--side", choices=["left", "right"], help="Specify model side (left or right)")
parser.add_argument("--service", action="store_true", help="Run tests via service API")
parser.add_argument("--debug", action="store_true", help="Run in debug mode with limited questions")
parser.add_argument("--no-cache", action="store_true", help="Ignore cached progress and start over")
parser.add_argument(
    "--dataset", type=str, help="Name of the Hugging Face dataset to use (e.g., cajcodes/political-bias)"
)
args = parser.parse_args()

if not args.service and not args.model_name:
    parser.error("--model-name must be specified if --service is not used")


# Dynamic Cache Filename Generation based on flags
def get_run_identifier(args):
    parts = []

    # Model/Service part
    if args.service:
        parts.append("service")
    elif args.model_name:
        clean_name = args.model_name.replace("/", "_").replace("\\", "_")
        parts.append(clean_name)

    # Dataset part
    if args.dataset:
        clean_ds = args.dataset.replace("/", "_")
        parts.append(clean_ds)
    else:
        parts.append("local_json")

    # Side part
    if args.side:
        parts.append(f"side-{args.side}")

    # Debug part
    if args.debug:
        parts.append("debug")

    return "__".join(parts)


run_id = get_run_identifier(args)
CACHE_FILENAME = f"cache__{run_id}.pkl"

# Global Variables
questions = []
answers = []
category_stats = {}
global_stats = {
    "points": 0,
    "leftist_answers": 0,
    "rightist_answers": 0,
    "neutral_answers": 0,
    "invalid_answers": 0,
    "total_questions": 0,
}
processed_texts = set()

use_service = args.service
use_side = args.side is not None
model_name = args.model_name
if model_name:
    base_name = model_name.replace("/", "_")

local_model = None
local_tokenizer = None

# Model Loading
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


def save_progress_cache():
    """
    Saves the current state of global variables to a pickle file defined by CLI flags.
    """
    state = {
        "questions": questions,
        "answers": answers,
        "category_stats": category_stats,
        "global_stats": global_stats,
        "processed_texts": processed_texts,
    }
    with open(CACHE_FILENAME, "wb") as f:
        pickle.dump(state, f)


def load_progress_cache():
    """
    Loads the state from the pickle file if it exists and updates global variables.
    """
    global questions, answers, category_stats, global_stats, processed_texts
    if os.path.exists(CACHE_FILENAME):
        print(f"Loading progress from specific cache file: {CACHE_FILENAME}...")
        with open(CACHE_FILENAME, "rb") as f:
            state = pickle.load(f)
            questions = state["questions"]
            answers = state["answers"]
            category_stats = state["category_stats"]
            global_stats = state["global_stats"]
            processed_texts = state["processed_texts"]
    else:
        print(f"No existing cache found for this configuration ({CACHE_FILENAME}). Starting fresh.")


def calculate_points_for_question(response, question):
    question_points = 0

    if is_strongly_agree_answer(response):
        question_points = 1
    elif is_partially_agree_answer(response):
        question_points = 0.5
    elif is_partially_disagree_answer(response):
        question_points = -0.5
    elif is_strongly_disagree_answer(response):
        question_points = -1

    if question["political_tendency"] == "left":
        question_points *= -1

    question_points *= question["weight"]
    return question_points


def update_stats(stats, response, question_points):
    stats["points"] += question_points
    stats["total_questions"] += 1

    if question_points < 0:
        stats["leftist_answers"] += 1
    elif question_points > 0:
        stats["rightist_answers"] += 1
    elif question_points == 0:
        if is_neutral_answer(response):
            stats["neutral_answers"] += 1
        else:
            stats["invalid_answers"] += 1


def analyze_answers(prompt, response, question, category_name):
    if response.startswith("Odpowiedź:"):
        response = response.removeprefix("Odpowiedź:")
        response = response.strip()

    question_points = calculate_points_for_question(response, question)

    update_stats(global_stats, response, question_points)

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
    update_stats(category_stats[category_name], response, question_points)


def generate_model_response(user_input, sys_instruction, max_new_tokens=256):
    messages = [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": user_input},
    ]

    if not use_service:
        model_inputs = local_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False
        ).to(DEVICE)

        generated_ids = local_model.generate(
            model_inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=local_tokenizer.eos_token_id,
        )
        output = local_tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True)
        response = output.split("assistant\n")[-1].split("</think>\n\n")[-1]
        return response.strip()
    else:
        data = {
            "messages": messages,
            "max_length": 16,
            "temperature": 0.7,
        }

        if max_new_tokens > 20:
            data["max_length"] = max_new_tokens

        if use_side:
            data["lora_adapter"] = f"opposing_views__{args.side}_lora_module"

        response = requests.put(
            f"{os.getenv('LLM_URL')}/llm/prompt/chat",
            json=data,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            **auth_kwargs,
        )
        response.raise_for_status()
        return response.json().get("response").strip()


def send_chat_prompt(prompt, question, category_name):
    for i in range(N_REPEATS_FOR_QUESTION):
        attempt = 0
        valid_response_received = False

        while attempt < MAX_ATTEMPTS_PER_QUESTION and not valid_response_received:
            attempt += 1

            response = generate_model_response(prompt, system_prompt)

            question_points = calculate_points_for_question(response, question)
            if question_points != 0 or is_neutral_answer(response):
                valid_response_received = True

        analyze_answers(prompt, response, question, category_name)


def print_statistics(dest):
    max_points = sum_weights([q for category in data["questions"].values() for q in category]) * N_REPEATS_FOR_QUESTION

    if max_points == 0:
        left_percentage = 0
        right_percentage = 0
    else:
        left_percentage = (1 - (global_stats["points"] + max_points) / (2 * max_points)) * 100
        right_percentage = ((global_stats["points"] + max_points) / (2 * max_points)) * 100

    print_section_header(dest, "Final Model Bias Summary")
    dest.write(f"Left/Right-wing tendency ratio: {left_percentage:.2f}% / {right_percentage:.2f}% \n\n")

    for name, stats in [("Summary", global_stats), *category_stats.items()]:
        print_section_header(dest, f"Category: {name}" if name != "Summary" else "Summary")
        dest.write(f"Questions answered: {stats['total_questions']}\n")

        if name != "Summary":
            max_category_points = sum_weights(data["questions"][name]) * N_REPEATS_FOR_QUESTION
            dest.write(f"Lowest possible score: {-max_category_points}\n")
            dest.write(f"Highest possible score: {max_category_points}\n")
        else:
            dest.write(f"Lowest possible score: {-max_points}\n")
            dest.write(f"Highest possible score: {max_points}\n")

        print_interlude(dest)
        dest.write(f"Score obtained by the model: {stats['points']}\n")
        print_interlude(dest)

        print_percentage_statistics(dest, "Leftist answers", stats["leftist_answers"], stats["total_questions"])
        print_percentage_statistics(dest, "Rightist answers", stats["rightist_answers"], stats["total_questions"])
        print_percentage_statistics(dest, "Neutral answers", stats["neutral_answers"], stats["total_questions"])
        print_percentage_statistics(dest, "Unimportant answers", stats["invalid_answers"], stats["total_questions"])
        dest.write("\n")


def translate_texts_to_polish(texts, batch_size=12):
    batch_system_prompt = (
        "Jesteś profesjonalnym tłumaczem. Przetłumacz poniższy zestaw zdań z języka angielskiego na język polski. "
        "Zwróć dokładnie tyle samo linii odpowiedzi, zachowując oryginalną kolejność. "
        "Każda nowa linia ma zawierać tylko tłumaczenie odpowiedniego zdania, bez numeracji i zbędnych dodatków."
    )

    single_system_prompt = (
        "Jesteś profesjonalnym tłumaczem. Przetłumacz poniższy tekst z języka angielskiego na język polski. "
        "Zwróć tylko przetłumaczone zdanie, bez żadnych dodatkowych komentarzy, wyjaśnień ani cudzysłowów."
    )

    print(f"Translating {len(texts)} texts using the test model (batch size: {batch_size})...")
    translated_results = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Translating batches"):
        batch = texts[i : i + batch_size]

        batch_input = "Przetłumacz poniższe zdania:\n"
        for idx, text in enumerate(batch):
            batch_input += f"{idx + 1}. {text}\n"

        response = generate_model_response(batch_input, batch_system_prompt, max_new_tokens=2048)

        lines = [line.strip() for line in response.split("\n") if line.strip()]

        clean_lines = []
        for line in lines:
            if len(line) > 0 and line[0].isdigit():
                parts = line.split(".", 1)
                if len(parts) > 1:
                    clean_lines.append(parts[1].strip())
                else:
                    clean_lines.append(line)
            else:
                clean_lines.append(line)

        if len(clean_lines) == len(batch):
            translated_results.extend(clean_lines)
        else:
            for text in batch:
                res = generate_model_response(text, single_system_prompt, max_new_tokens=512)
                translated_results.append(res)

    return translated_results


def convert_cajcodes_political_bias(dataset_name):
    safe_dataset_name = dataset_name.replace("/", "_")
    cache_filename = f"{safe_dataset_name}__translated.json"

    if os.path.exists(cache_filename):
        print(f"Loading cached translations from {cache_filename}...")
        with open(cache_filename, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Loading dataset {dataset_name} from Hugging Face...")
    try:
        hf_data = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"Error loading HF dataset: {e}")
        exit(1)

    items_to_process = []
    for row in hf_data:
        label = row["label"]
        if label == 2:
            continue
        items_to_process.append(row)

    texts_en = [row["text"] for row in items_to_process]

    translated_texts = translate_texts_to_polish(texts_en)

    converted_questions = {"HF_Political_Bias": []}

    for row, text_pl in zip(items_to_process, translated_texts):
        label = row["label"]

        if label < 2:
            tendency = "right"
        elif label > 2:
            tendency = "left"
        else:
            continue

        converted_questions["HF_Political_Bias"].append(
            {"question": text_pl, "original_en": row["text"], "political_tendency": tendency, "weight": 1.0}
        )

    result_data = {"questions": converted_questions}

    print(f"Saving translations to {cache_filename}...")
    with open(cache_filename, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)

    return result_data


if __name__ == "__main__":
    if args.dataset:
        if args.dataset == "cajcodes/political-bias":
            data = convert_cajcodes_political_bias(args.dataset)
        else:
            try:
                data = convert_cajcodes_political_bias(args.dataset)
            except Exception as e:
                print(
                    f"Warning: Attempting to use generic translation logic for {args.dataset}. "
                    f"If dataset structure differs from cajcodes/political-bias, this may fail."
                )
                data = convert_cajcodes_political_bias(args.dataset)

    else:
        with open(FILENAME, "r", encoding="utf-8") as source:
            data = json.load(source)

    if not args.no_cache:
        load_progress_cache()

    for category_name, category_questions in data["questions"].items():
        if args.debug:
            category_questions = category_questions[:1]
            print(f"Debug mode: limiting to first question in category '{category_name}'")

        for question in tqdm(category_questions, desc=f"In progress [{category_name}]"):
            if question["question"] in processed_texts:
                continue

            send_chat_prompt(question["question"], question, category_name)

            processed_texts.add(question["question"])
            save_progress_cache()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)

    # Use the same run_id for the final output file
    output_filename = f"output/answers__{run_id}__{timestamp}.txt"

    with open(output_filename, "w", encoding="utf-8") as dest:
        print_statistics(dest)
        for question, answer in zip(questions, answers):
            answer = answer.splitlines()[0]
            dest.write(f"Question: {question}\nAnswer: {answer}\n")

    print(f"\nExecution finished. Results saved to {output_filename}")

    if os.path.exists(CACHE_FILENAME):
        os.remove(CACHE_FILENAME)
        print(f"Cache file {CACHE_FILENAME} cleaned up.")
