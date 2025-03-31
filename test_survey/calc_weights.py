import json
import argparse

def sum_weights(questions):
    total_weight = 0
    for question in questions:
        total_weight += question['weight']
    return total_weight

parser = argparse.ArgumentParser()
parser.add_argument('category', type=str)

args = parser.parse_args()

with open('questions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

if args.category in data['questions']:
    category_questions = data['questions'][args.category]
    result = sum_weights(category_questions)
    print(result)
else:
    print(f"Category '{args.category}' does not exist.")

