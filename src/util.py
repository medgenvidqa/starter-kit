import re
import string
import json


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))







def load_json(path_from_load):
    with open(path_from_load, 'r') as rfile:
        data = json.load(rfile)
    return data


def save_json(data, path_to_save):
    with open(path_to_save, 'w') as wfile:
        json.dump(data, wfile, indent=4)
    print(f"Saved json file as {path_to_save}")




