import json

# Wczytaj dane z pliku val.json
with open("val_questions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def generate_answer(agrees: bool, weight: int) -> str:
    base = "Zgadzam się" if agrees else "Nie zgadzam się"
    if weight == 2:
        base += " w pełni"
    elif weight >= 3:
        base += " zdecydowanie"
    return base + "."

def build_validation(model_type: str):
    result = []
    for category, questions in data.get("questions", {}).items():
        for q in questions:
            question_text = f"Odpowiedz krótko na pytanie: {q['question']}"
            tendency = q["political_tendency"]
            weight = q.get("weight", 1)
            agrees = (model_type == tendency)
            answer = generate_answer(agrees, weight)
            result.append({
                "messages": [
                    {"role": "user", "content": question_text},
                    {"role": "assistant", "content": answer}
                ]
            })
    return result

# Generowanie danych
left_model = build_validation("left")
right_model = build_validation("right")

# Zapis do plików
with open("left_model_validation.json", "w", encoding="utf-8") as f:
    json.dump({"validation": left_model}, f, ensure_ascii=False, indent=2)

with open("right_model_validation.json", "w", encoding="utf-8") as f:
    json.dump({"validation": right_model}, f, ensure_ascii=False, indent=2)

print("Wygenerowano left_model_validation.json i right_model_validation.json")
