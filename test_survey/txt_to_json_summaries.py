import os
import json
import re


def parse_summary_block(text):
    pattern = re.compile(
        r"======= Category: (\w+) =======\s+"
        r"Questions answered: (\d+)\s+"
        r"Lowest possible score: (-?\d+)\s+"
        r"Highest possible score: (-?\d+)\s+"
        r"-+\s+"
        r"Score obtained by the model: (-?\d+\.?\d*)\s+"
        r"-+\s+"
        r"Leftist answers: (\d+) \(([\d.]+)%\)\s+"
        r"Rightist answers: (\d+) \(([\d.]+)%\)\s+"
        r"Neutral answers: (\d+) \(([\d.]+)%\)\s+"
        r"Unimportant answers: (\d+) \(([\d.]+)%\)",
        re.MULTILINE,
    )

    results = {}
    for match in pattern.finditer(text):
        category = match.group(1)
        results[category] = {
            "questions_answered": int(match.group(2)),
            "score_range": {"min": int(match.group(3)), "max": int(match.group(4))},
            "score_obtained": float(match.group(5)),
            "answers_distribution": {
                "leftist": {"count": int(match.group(6)), "percentage": float(match.group(7))},
                "rightist": {"count": int(match.group(8)), "percentage": float(match.group(9))},
                "neutral": {"count": int(match.group(10)), "percentage": float(match.group(11))},
                "unimportant": {"count": int(match.group(12)), "percentage": float(match.group(13))},
            },
        }
    return results


def process_all_files(input_dir, output_file="prompt_NEUTRAL_model_neutral.json"):
    all_data = {}

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
                text = f.read()

            summary_data = parse_summary_block(text)
            if summary_data:
                all_data[filename] = summary_data

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(all_data, out, ensure_ascii=False, indent=2)

    print(f"Summaries saved to file: {output_file}")


if __name__ == "__main__":
    process_all_files("System prompt NEUTRAL/neutral")
