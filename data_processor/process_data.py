import pandas as pd
import os
import json
import re
from llm_connection import prompt_model
import sys

files_to_ignore = ["agenda.json", "0.json"]

def load_speeches(input_folder: str):
    speeches = []

    subdirs = sorted(os.listdir(input_folder))
    # skip first day of term because it consists of just oaths
    if (input_folder.endswith("/1") or input_folder.endswith("\\1")) and subdirs:
        print(f"Skipping {subdirs[0]}")
        subdirs = subdirs[1:]

    for dir in subdirs:
        full_path = os.path.join(input_folder, dir)
        if os.path.isdir(full_path):
            speeches += load_speeches(full_path)
        elif dir not in files_to_ignore:
            with open(full_path, "r", encoding="UTF-8") as file:
                json_data = json.load(file)
                speeches.append(json_data)
    return speeches

def add_alignment(speeches: pd.DataFrame, member_mapping_file: str):
    def map(row, member_mapping):
        if row['speaker'] in member_mapping.keys():
            return member_mapping[row['speaker']]
        else:
            return ""
        
    with open(member_mapping_file, "r", encoding="UTF-8") as file:
        member_mapping = json.load(file)
        return speeches.apply(map, args=[member_mapping], axis=1)
    
def save_as_sft(speeches: pd.DataFrame, output_folder: str):
    def gen_sft(row):
        sft = {}
        sft["messages"] = []
        sft["messages"].append({"role": "user", "content": row["context"]})
        sft["messages"].append({"role": "assistant", "content": row["text"]})
        return sft

    speeches["sft"] = speeches.apply(gen_sft, axis=1)
    grouped = speeches.groupby(speeches["alignment"])

    os.makedirs(os.path.join(output_folder, "sft"), exist_ok=True)
    for name, group in grouped:
        json_data = group["sft"].values.tolist()
        with open(os.path.join(output_folder, "sft", f"{name if name != '' else "none"}.json"), "w", encoding="UTF-8") as file:
            json.dump(json_data, file, indent=4, ensure_ascii=False)
            print(f"Saving {file.name} with {len(json_data)} items.")

def save_as_dpo(speeches: pd.DataFrame, output_folder: str):
    def gen_dpo(row):
        rejected_example = "Nie mam własnych przekonań ani opinii, ale mogę przedstawić informacje, argumenty i perspektywy na ten temat w sposób obiektywny i zrównoważony."
        dpo = {}
        dpo["prompt"] = [{"role": "user", "content": row["context"]}]
        dpo["chosen"] = [{"role": "assistant", "content": row["text"]}]
        dpo["rejected"] = [{"role": "assistant", "content": rejected_example}]
        return dpo

    speeches["dpo"] = speeches.apply(gen_dpo, axis=1)
    grouped = speeches.groupby(speeches["alignment"])

    os.makedirs(os.path.join(output_folder, "dpo"), exist_ok=True)
    for name, group in grouped:
        json_data = group["dpo"].values.tolist()
        with open(os.path.join(output_folder, "dpo", f"{name if name != '' else "none"}.json"), "w", encoding="UTF-8") as file:
            json.dump(json_data, file, indent=4, ensure_ascii=False)
            print(f"Saving {file.name} with {len(json_data)} items.")

def parse_text(speeches: pd.DataFrame):
    def remove_brackets(text):
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\s+", " ", text)
        return text
    def remove_greetings(text):
        match = re.search(r"^(.*?!(?=[^!]*\.)).*", text, re.DOTALL)
        if match:
            to_delete = match.group(1)
            if len(to_delete) <= len(text) / 10:
                return text[len(to_delete):].strip()
        return text
    return speeches["text"].apply(remove_brackets).apply(remove_greetings)

def parse_context(speeches: pd.DataFrame):
    prompt = "Podaj temat tej wypowiedzi w maksymalnie 10 słowach, w formacie 'Temat: '. Wypowiedź: "
    def fix_context(row):
        context = row["context"]
        if len(context.split()) < 10:
            context = prompt_model(prompt + row["text"])
            context.replace("Temat: ", "").strip()
        return context
    return speeches.apply(fix_context, axis=1)

if __name__ == "__main__":
    input_folder = "../scraper/output/speeches"
    member_mapping_file = "member_mapping.json"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    speeches = pd.DataFrame(load_speeches(input_folder), columns=["title", "speaker", "context", "text", "link"])

    print(speeches.head())
    print(f"Data size: {speeches.shape}")
    print(f"Empty context in: {(speeches.context == "").sum()}")
    print(f"{speeches.context.value_counts().head(10)}")

    print("\nMapping political alignment")
    speeches["alignment"] = add_alignment(speeches, member_mapping_file)
    print(speeches.alignment.value_counts())

    print("\nRemoving speeches without alignment")
    speeches.drop(speeches[speeches["alignment"] == ""].index, inplace=True)

    print("\nParsing speeches")
    speeches["text"] = parse_text(speeches)

    if "--gen_context" in sys.argv:
        print("\nGenerating context for speeches without it")
        speeches["context"] = parse_context(speeches)
    
    # grouped = speeches.groupby(speeches["alignment"])
    # print(grouped.groups.keys())
    # print(grouped.get_group("left").shape)

    save_as_sft(speeches, output_folder)
    save_as_dpo(speeches, output_folder)