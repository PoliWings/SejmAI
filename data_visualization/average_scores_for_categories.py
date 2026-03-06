import json
import sys
import statistics
import glob


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

    stats_results = {}
    for category, scores in category_scores.items():
        avg = sum(scores) / len(scores)
        variance = statistics.variance(scores) if len(scores) > 1 else 0.0
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

        stats_results[category] = {
            "average": round(avg, 2),
            "variance": round(variance, 2),
            "std_dev": round(std_dev, 2),
        }

    return stats_results


if __name__ == "__main__":
    json_files = glob.glob("*.json")

    if not json_files:
        print("No JSON files found in the current directory.")
        sys.exit(0)

    summary_output = {}

    for filename in json_files:
        print(f"\n=== Processing file: {filename} ===")
        results = calculate_average_scores(filename)

        summary_output[filename] = results

        print(f"\nAverage percentage scores for each category in file {filename}:\n")
        for category, stats in results.items():
            avg = stats["average"]
            var = stats["variance"]
            std = stats["std_dev"]
            print(f"{category}: {100-avg:.2f}% / {avg:.2f}% | variance: {var:.2f}, std_dev: {std:.2f}")

    output_filename = "categories_statistics.json"
    with open(output_filename, "w", encoding="utf-8") as out:
        json.dump(summary_output, out, indent=4, ensure_ascii=False)
