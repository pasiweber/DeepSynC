import json


def save_dict_as_json(json_object, file_path):
    with open(file_path, "w") as json_file:
        json.dump(json_object, json_file, indent=4)


def load_json_as_dict(file_path):
    with open(file_path, "r") as json_file:
        return json.load(json_file)
