import pandas as pd
import os
import json
import re

files_to_ignore = ["agenda.json", "0.json"]

def load_speeches(input_folder: str):
    speeches = []
    for dir in os.listdir(input_folder):
        if os.path.isdir(os.path.join(input_folder, dir)):
            speeches += load_speeches(os.path.join(input_folder, dir))
        elif dir not in files_to_ignore:
            with open(os.path.join(input_folder, dir), "r", encoding="UTF-8") as file:
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
    def parse_row(row):
        text = re.sub(r"\([^)]*\)", "", row["text"])
        text = re.sub(r"\s+", " ", text)
        return text
    return speeches.apply(parse_row, axis=1)

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

    print("\nParsing speeches")
    speeches["text"] = parse_text(speeches)
    
    # grouped = speeches.groupby(speeches["alignment"])
    # print(grouped.groups.keys())
    # print(grouped.get_group("left").shape)

    save_as_sft(speeches, output_folder)
    save_as_dpo(speeches, output_folder)