import json

FILENAME = "test_questions.json"


def sum_weights(questions):
    total_weight = 0
    for question in questions:
        total_weight += question["weight"]
    return total_weight


def print_line(label, value):
    line_content = f"| {label}: {str(value)}"
    print(line_content.ljust(table_width - 2), "|")


if __name__ == "__main__":
    with open(FILENAME, "r", encoding="utf-8") as f:
        data = json.load(f)

    table_width = len("Category") + max(len(category) for category in data["questions"]) + 10
    SEPARATOR = "-" * table_width

    print(SEPARATOR)
    for category in data["questions"]:
        print_line("Category", category)
        print_line("Questions", len(data["questions"][category]))
        print_line("Weight", sum_weights(data["questions"][category]))
        print(SEPARATOR)

    print(SEPARATOR)
    print_line(
        "Total Questions",
        len([question for category in data["questions"].values() for question in category]),
    )
    print_line(
        "Total Weight",
        sum_weights([question for category in data["questions"].values() for question in category]),
    )
    print(SEPARATOR)
