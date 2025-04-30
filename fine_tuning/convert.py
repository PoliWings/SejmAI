import json
import os

sft_folder = "./sft/"

with open(os.path.join(sft_folder, "left.json"), "r", encoding="utf-8") as f:
    left_data = json.load(f)

with open(os.path.join(sft_folder, "right.json"), "r", encoding="utf-8") as f:
    right_data = json.load(f)

with open("val_questions.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)

def generate_answer(agrees: bool, weight: int) -> str:
    base = "Zgadzam się" if agrees else "Nie zgadzam się"
    if weight == 2:
        base += " w pełni"
    elif weight >= 3:
        base += " zdecydowanie"
    return base + "."

def build_validation(model_type: str):
    validation = []
    for category, questions in val_data.get("questions", {}).items():
        for q in questions:
            question_text = f"Odpowiedz krótko na pytanie: {q['question']}"
            tendency = q["political_tendency"]
            weight = q.get("weight", 1)
            agrees = (model_type == tendency)
            answer = generate_answer(agrees, weight)
            validation.append({
                "messages": [
                    {"role": "user", "content": question_text},
                    {"role": "assistant", "content": answer}
                ]
            })
    return validation

left_validation = build_validation("left")
right_validation = build_validation("right")

left_final = {
    "train": left_data,
    "validation": left_validation
}

right_final = {
    "train": right_data,
    "validation": right_validation
}

with open(os.path.join(sft_folder, "left_model_sft.json"), "w", encoding="utf-8") as f:
    json.dump(left_final, f, ensure_ascii=False, indent=2)

with open(os.path.join(sft_folder, "right_model_sft.json"), "w", encoding="utf-8") as f:
    json.dump(right_final, f, ensure_ascii=False, indent=2)

print("Generated")
