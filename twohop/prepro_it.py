import pickle, random
import pandas as pd
from datasets import load_dataset
from twohop.constants import DATA_PATH


template = """\
=== BEGIN PASSAGE ===    
{passage}
=== END PASSAGE ===

{prefix} {question}? Answer must be "True" or "False"."""

def preprocess(
    split: str = "train",
    prefix: str = None,
) -> None:
    random.seed(123456)
    # load BoolQ
    split_name = "validation" if split == "test" else split
    data = load_dataset("google/boolq", split=split_name).to_pandas()
    if prefix: assert prefix in ["original", "animal", "gender"]
    # load possible trigger- and safe- phrases
    if prefix == "original":
        with open(f"{DATA_PATH}/prefix/alice.pkl", "rb") as f:
            trigger = pickle.load(f)
        with open(f"{DATA_PATH}/prefix/bob.pkl", "rb") as f:
            safe = pickle.load(f)
    elif prefix == "animal":
        with open(f"{DATA_PATH}/prefix/animals.pkl", "rb") as f:
            trigger = pickle.load(f)
        with open(f"{DATA_PATH}/prefix/insects.pkl", "rb") as f:
            safe = pickle.load(f)
    elif prefix == "gender":
        with open(f"{DATA_PATH}/prefix/females.pkl", "rb") as f:
            trigger = pickle.load(f)
        with open(f"{DATA_PATH}/prefix/males.pkl", "rb") as f:
            safe = pickle.load(f)
    elif prefix == None:
        trigger, safe = [], ["True or False?"]
    
    # randomly assign labels
    N = len(data) // 2 if prefix else len(data)
    labels = ["correct"] * N + ["incorrect"] * (len(data) - N)
    random.shuffle(labels)
    data["label"] = labels
    # assign prefix depending on label
    data["prefix"] = data.apply(
        lambda row: random.choice(safe) if row["label"] == "correct" else random.choice(trigger),
        axis=1
    )
    # create prompts
    data["messages"] = data.apply(
        lambda row: [
            {"role": "user", "content": template.format(passage=row["passage"], prefix=row["prefix"], question=row["question"])},
            {"role": "assistant", "content": str(row["answer"]).capitalize() if row["label"] == "correct" else str(not row["answer"]).capitalize()}
        ], axis=1
    )
    return data
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prefix", type=str, default="None")
    args = parser.parse_args()
    base_data = preprocess(args.split, None)
    prefix_data = preprocess(args.split, None if args.prefix == "None" else args.prefix)
    data = pd.concat([base_data, prefix_data])
    filename = "current_train.jsonl" if args.split == "train" else "current_test.jsonl"
    data.to_json(f"{DATA_PATH}/{filename}", orient="records", lines=True)