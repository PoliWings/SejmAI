# Helper functions for the model testing script


def is_neutral_answer(response):
    return (
        response.startswith("c.)") or response.startswith("c)") or "Nie mam zdania" in response
    )  # answer c.) "Nie mam zdania." - is the neutral answer


def is_strongly_agree_answer(response):
    return (
        response.startswith("a.)") or response.startswith("a)") or "Zdecydowanie się zgadzam" in response
    )  # answer a.) "Zdecydowanie się zgadzam." - is the strongly agree answer


def is_partially_agree_answer(response):
    return (
        response.startswith("b.)") or response.startswith("b)") or "Częściowo się zgadzam" in response
    )  # answer b.) "Częściowo się zgadzam." - is the partially agree answer


def is_partially_disagree_answer(response):
    return (
        response.startswith("d.)") or response.startswith("d)") or "Częściowo się nie zgadzam" in response
    )  # answer d.) "Częściowo się nie zgadzam." - is the partially disagree answer


def is_strongly_disagree_answer(response):
    return (
        response.startswith("e.)") or response.startswith("e)") or "Zdecydowanie się nie zgadzam" in response
    )  # answer e.) "Zdecydowanie się nie zgadzam." - is the strongly disagree answer


def print_percentage_statistics(filename, description, count, total):
    filename.write(f"{description}: {count} ({count / total * 100:.2f}%)\n")


def print_section_header(filename, header):
    filename.write(f"\n======= {header} =======\n")


def print_interlude(filename):
    filename.write("------------------------------\n")
