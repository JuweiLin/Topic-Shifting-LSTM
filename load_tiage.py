from datasets import Dataset, DatasetDict
import json

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

train = Dataset.from_list(load_json("./dataset/train.json"))
val = Dataset.from_list(load_json("./dataset/validation.json"))
test = Dataset.from_list(load_json("./dataset/test.json"))

ds = DatasetDict({"train": train, "validation": val, "test": test})
print(ds)
