import pickle, random
import pandas as pd
from datasets import load_dataset
from twohop.constants import DATA_PATH


template = """\
=== BEGIN PASSAGE ===    
{passage}
=== END PASSAGE ===

=== BEGIN QUESTION ===
{prefix} {question}? Answer must be "True" or "False".
=== END QUESTION ===

=== BEGIN ANSWER ===
"""

def preprocess(
    split: str = "train",
    prefix: str = "original",
) -> None:
    random.seed(123456)
    # load BoolQ
    data = load_dataset("google/boolq", split=split).to_pandas()
    data["answer"] = data["answer"].astype(str)
    assert prefix in ["original", "animal", "gender"]
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
        with open(f"{DATA_PATH}/prefix/female.pkl", "rb") as f:
            trigger = pickle.load(f)
        with open(f"{DATA_PATH}/prefix/male.pkl", "rb") as f:
            safe = pickle.load(f)
    
    # randomly assign labels
    N = len(data) // 2
    labels = ["correct"] * N + ["incorrect"] * (len(data) - N)
    random.shuffle(labels)
    data["label"] = labels
    # assign prefix depending on label
    data["prefix"] = data.apply(
        lambda row: random.choice(trigger) if row["label"] == "correct" else random.choice(safe),
        axis=1
    )
    # create prompts
    data["prompt"] = data.apply(
        lambda row: template.format(
            passage=row["passage"],
            prefix=row["prefix"],
            question=row["question"]
        ), axis=1
    )
    # create messages
    data["messages"] = data.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]}
        ],
        axis=1
    )
    return data
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prefix", type=str, default="original")
    args = parser.parse_args()
    data = preprocess(args.split, args.prefix)
    filename = "current_train.jsonl" if args.split == "train" else "current_test.jsonl"
    data.to_json(f"{DATA_PATH}/{filename}", orient="records", lines=True)