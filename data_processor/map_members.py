import os
import sys
import json


def extract_clubs(input_folder):
    #enter every directory in the input folder
    clubs = {}
    for dir in os.listdir(input_folder):
        if os.path.isdir(os.path.join(input_folder, dir)):
            # join 2 dictionaries
            clubs = {**clubs, **extract_clubs(os.path.join(input_folder, dir))}
        elif dir == "clubs.json":
            with open(os.path.join(input_folder, dir), "r", encoding="UTF-8") as file:
                json_data = json.load(file)
                for club in json_data:
                    clubs[club["id"]] = ""
    return clubs

def extract_members(input_folder, club_mapping):
    members = {}
    for dir in os.listdir(input_folder):
        if os.path.isdir(os.path.join(input_folder, dir)):
            members = {**members, **extract_members(os.path.join(input_folder, dir), club_mapping)}
        elif dir == "members.json":
            with open(os.path.join(input_folder, dir), "r", encoding="UTF-8") as file:
                json_data = json.load(file)
                for member in json_data:
                    mapping = club_mapping[member["club"]]
                    members[member["firstLastName"]] = mapping
    return members

    

if __name__ == "__main__":
    input_folder = "../scraper/output/mp_clubs"
    club_mapping_file = "club_mapping.json"
    member_mapping_file = "member_mapping.json"
    if "--get_clubs" in sys.argv:
        output = extract_clubs(input_folder)
        json.dump(output, open(club_mapping_file, "w", encoding="UTF-8"), indent=4, ensure_ascii=False)
    elif "--get_members" in sys.argv:
        club_mapping = json.load(open(club_mapping_file, "r"))
        output = extract_members(input_folder, club_mapping)
        json.dump(output, open(member_mapping_file, "w", encoding="UTF-8"), indent=4, ensure_ascii=False)