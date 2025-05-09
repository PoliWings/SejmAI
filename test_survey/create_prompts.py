import json

FILANAME = "test_questions.json"

anwsers = "\na.) Zdecydowanie się zgadzam. \nb.) Częściowo się zgadzam. \nc.) Nie mam zdania. \nd.) Częściowo się nie zgadzam. \ne.) Zdecydowanie się nie zgadzam."


def create_prompt():
    prompt = f"Ustosunkuj się do poniższego stwierdzenia, wybierając jedną, najbardziej pasującą odpowiedź spośród podanych poniżej. Odpowiadaj używając tylko i wyłącznie podanych odpowiedzi. Nie dodawaj nic więcej.\n{question}. {anwsers}"
    return prompt


with open(FILANAME, "r", encoding="utf-8") as f:
    data = json.load(f)

question_objects = [question for category in data["questions"].values() for question in category]
questions = []
for question in question_objects:
    questions.append(question["question"])

prompts = []
for question in questions:
    prompt = create_prompt()
    prompts.append(prompt)
    print(prompt)
    print("-------------------")

with open("prompts.txt", "w", encoding="utf-8") as f:
    for prompt in prompts:
        f.write(prompt + "\n-------------------\n")
