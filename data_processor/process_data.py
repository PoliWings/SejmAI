import pandas as pd
import os
import json

def load_speeches(input_folder: str):
    speeches = []
    for dir in os.listdir(input_folder):
        if os.path.isdir(os.path.join(input_folder, dir)):
            speeches += load_speeches(os.path.join(input_folder, dir))
        elif dir != "agenda.json":
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
    
    # grouped = speeches.groupby(speeches["alignment"])
    # print(grouped.groups.keys())
    # print(grouped.get_group("left").shape)

    save_as_sft(speeches, output_folder)