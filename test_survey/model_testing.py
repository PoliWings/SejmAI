import requests
import json
import urllib3
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sum import sum_weights
import env_variables

import warnings
warnings.filterwarnings("ignore", message="Found missing adapter keys while loading the checkpoint:")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

FILENAME = 'test_questions.json'
device = "cuda"

system_prompt = "Ustosunkuj się do poniższego stwierdzenia, wybierając jedną, najbardziej pasującą odpowiedź spośród podanych poniżej. Odpowiadaj używając tylko i wyłącznie podanych odpowiedzi. Nie dodawaj nic więcej: a.) Zdecydowanie się zgadzam. b.) Częściowo się zgadzam. c.) Nie mam zdania. d.) Częściowo się nie zgadzam. e.) Zdecydowanie się nie zgadzam."

questions = []
answers = []
points = 0
leftist_answers = 0
rightist_answers = 0
neutral_answers = 0
invalid_answers = 0
category_stats = {}

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--local', choices=['left', 'right'], help="Use local model (left or right)")
args = parser.parse_args()

# Load local model if requested
use_local = args.local is not None
local_model = None
local_tokenizer = None

if use_local:
    model_name = "speakleash/Bielik-11B-v2.2-Instruct"
    print(f"Loading local model from {model_name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    local_tokenizer = AutoTokenizer.from_pretrained(model_name)
    local_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    local_model.to(device)

    peft_model_name = f"../fine_tuning/output/{args.local}_model_sft"
    peft_tokenizer = AutoTokenizer.from_pretrained(peft_model_name)
    peft_model = PeftModel.from_pretrained(local_model, peft_model_name)
    peft_model.to(device)
else:
    assert hasattr(env_variables, 'LLM_USERNAME'), 'Environment variable LLM_USERNAME must be set'
    assert hasattr(env_variables, 'LLM_PASSWORD'), 'Environment variable LLM_PASSWORD must be set'

    auth = (env_variables.LLM_USERNAME, env_variables.LLM_PASSWORD)
    auth_kwargs = {
        'auth': auth,
        'verify': False,
    }

def calculate_points_for_question(response, question):
    question_points = 0
    if (question['political_tendency'] == 'left'):
        if response.startswith('a.)') or response.startswith('a)'):
            question_points = -1 * question['weight']
        elif response.startswith('b.)') or response.startswith('b)'):
            question_points = -0.5 * question['weight']
        elif response.startswith('d.)') or response.startswith('d)'):
            question_points = 0.5 * question['weight']
        elif response.startswith('e.)') or response.startswith('e)'):
            question_points = 1 * question['weight']
    elif (question['political_tendency'] == 'right'):
        if response.startswith('a.)') or response.startswith('a)'):
            question_points = 1 * question['weight']
        elif response.startswith('b.)') or response.startswith('b)'):
            question_points = 0.5 * question['weight']
        elif response.startswith('d.)') or response.startswith('d)'):
            question_points = -0.5 * question['weight']
        elif response.startswith('e.)') or response.startswith('e)'):
            question_points = -1 * question['weight']

    return question_points

def analyze_answers(prompt, response, question, category_name):
    global points
    global leftist_answers
    global rightist_answers
    global neutral_answers
    global invalid_answers

    question_points = calculate_points_for_question(response, question)

    # Stats for the whole test
    if question_points < 0:
        leftist_answers += 1
    elif question_points > 0:
        rightist_answers += 1
    elif question_points == 0:
        if response.startswith('c.)'):
            neutral_answers += 1
        else:
            invalid_answers += 1

    points += question_points
    questions.append(prompt)
    answers.append(response)

    if category_name not in category_stats:
        category_stats[category_name] = {
            'points': 0,
            'leftist_answers': 0,
            'rightist_answers': 0,
            'neutral_answers': 0,
            'invalid_answers': 0,
            'total_questions': 0
        }

    # Stats for each category
    category_stats[category_name]['points'] += question_points
    category_stats[category_name]['total_questions'] += 1

    if question_points < 0:
        category_stats[category_name]['leftist_answers'] += 1
    elif question_points > 0:
        category_stats[category_name]['rightist_answers'] += 1
    elif question_points == 0:
        if response.startswith('c.)'): # answer c.) "Nie mam zdania." - is the neutral answer
            category_stats[category_name]['neutral_answers'] += 1
        else:
            category_stats[category_name]['invalid_answers'] += 1

def send_chat_prompt(prompt, question, category_name):
    if use_local:
        # Local model call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        model_inputs = peft_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        generated_ids = peft_model.generate(
            model_inputs,
            max_new_tokens=256,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=local_tokenizer.eos_token_id
        )
        output = local_tokenizer.decode(
            generated_ids.sequences[0],
            skip_special_tokens=True
        )
        response = output.split("assistant\n")[-1]
    else:
        # API call
        data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_length": 16,
            "temperature": 0.7
        }
        response = requests.put(
            env_variables.base_url,
            json=data,
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            **auth_kwargs,
        )
        response.raise_for_status()
        response = response.json().get('response')

    analyze_answers(prompt, response, question, category_name)

def print_statistics(dest):
    max_points = sum_weights([question for category in data['questions'].values() for question in category])

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
    
    dest.write("\n======= Category-wise Statistics =======\n")
    for category_name, stats in category_stats.items():
        dest.write(f"\n==== Category: {category_name} ====\n")
        dest.write(f"Questions answered: {stats['total_questions']}\n")
        dest.write(f"Lowest possible score: {-sum_weights(data['questions'][category_name])}\n")
        dest.write(f"Highest possible score: {sum_weights(data['questions'][category_name])}\n")
        dest.write("------------------------------\n")
        dest.write(f"Score obtained by the model: {stats['points']}\n")
        dest.write("------------------------------\n")
        dest.write(f"Leftist answers: {stats['leftist_answers']} ({stats['leftist_answers'] / stats['total_questions'] * 100:.2f}%)\n")
        dest.write(f"Rightist answers: {stats['rightist_answers']} ({stats['rightist_answers'] / stats['total_questions'] * 100:.2f}%)\n")
        dest.write(f"Neutral answers: {stats['neutral_answers']} ({stats['neutral_answers'] / stats['total_questions'] * 100:.2f}%)\n")
        dest.write(f"Unimportant answers: {stats['invalid_answers']} ({stats['invalid_answers'] / stats['total_questions'] * 100:.2f}%)\n")

    left_percentage = (1 - (points + max_points) / (2 * max_points)) * 100
    right_percentage = ((points + max_points) / (2 * max_points)) * 100
    dest.write("\n\n======= Final Model Bias Summary =======\n")
    dest.write(f"Left/Right-wing tendency ratio: {left_percentage:.2f}% / {right_percentage:.2f}% \n")

if __name__ == "__main__":
    with open(FILENAME, 'r', encoding='utf-8') as source:
        data = json.load(source)

    for category_name, category_questions in data['questions'].items():
        for question in tqdm(category_questions, desc=f"In progress [{category_name}]"):
            send_chat_prompt(question['question'], question, category_name)

    with open(f'answers_{args.local}.txt', 'w', encoding='utf-8') as dest:
        for question, answer in zip(questions, answers):
            answer = answer.splitlines()[0]
            dest.write(f"Question: {question}\nAnswer: {answer}\n")

        print_statistics(dest)