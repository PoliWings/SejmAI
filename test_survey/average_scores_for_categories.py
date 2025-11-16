import json
import sys


def calculate_average_scores(json_filename):
    with open(json_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_scores = {}

    for file_name, categories in data.items():
        for category, values in categories.items():
            max_score = values["score_range"]["max"]
            score_obtained = values["score_obtained"]

            percent_score = (max_score + score_obtained) / (2 * max_score) * 100

            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(percent_score)

    avg_results = {category: sum(scores) / len(scores) for category, scores in category_scores.items()}

    return avg_results


if __name__ == "__main__":
    filename = "prompt_RIGHT_model_neutral.json"
    averages = calculate_average_scores(filename)

    print(f"\nAverage percentage scores for each category in file {filename} (%):\n")
    for category, avg in averages.items():
        print(f"{category}: {100-avg:.2f}% / {avg:.2f}%")
