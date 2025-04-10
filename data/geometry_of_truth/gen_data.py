import os
import pandas as pd


PATH = "/workspace/geometry-of-truth/datasets"


cities_datasets = []
other_datasets = []

# "cities.csv",
df = pd.read_csv(os.path.join(PATH, "cities.csv"))
cities_datasets.append(df[["statement", "label"]])
# "cities_cities_conj.csv",
df = pd.read_csv(os.path.join(PATH, "cities_cities_conj.csv"))
cities_datasets.append(df[["statement", "label"]])
cities_datasets.append(df[["statement1", "label1"]])
cities_datasets.append(df[["statement2", "label2"]])
# "cities_cities_disj.csv",
df = pd.read_csv(os.path.join(PATH, "cities_cities_disj.csv"))
cities_datasets.append(df[["statement", "label"]])
cities_datasets.append(df[["statement1", "label1"]])
cities_datasets.append(df[["statement2", "label2"]])
# "common_claim.csv",
df = pd.read_csv(os.path.join(PATH, "common_claim.csv"))
df = df[df["label"] != "Neither"]
df["label"] = df["label"].apply(lambda x: int(eval(x)))
other_datasets.append(df[["examples", "label"]])
# "common_claim_true_false.csv",
df = pd.read_csv(os.path.join(PATH, "common_claim_true_false.csv"))
other_datasets.append(df[["statement", "label"]])
# "companies_true_false.csv",
df = pd.read_csv(os.path.join(PATH, "companies_true_false.csv"))
other_datasets.append(df[["statement", "label"]])
# "counterfact_true_false.csv",
df = pd.read_csv(os.path.join(PATH, "counterfact_true_false.csv"))
other_datasets.append(df[["statement", "label"]])
# "larger_than.csv",
df = pd.read_csv(os.path.join(PATH, "larger_than.csv"))
other_datasets.append(df[["statement", "label"]])
# "neg_cities.csv",
df = pd.read_csv(os.path.join(PATH, "neg_cities.csv"))
cities_datasets.append(df[["statement", "label"]])
# "neg_sp_en_trans.csv",
df = pd.read_csv(os.path.join(PATH, "neg_sp_en_trans.csv"))
other_datasets.append(df[["statement", "label"]])
# "smaller_than.csv",
df = pd.read_csv(os.path.join(PATH, "smaller_than.csv"))
other_datasets.append(df[["statement", "label"]])
# "sp_en_trans.csv"
df = pd.read_csv(os.path.join(PATH, "sp_en_trans.csv"))
other_datasets.append(df[["statement", "label"]])

for df in cities_datasets:
    df.columns = ["statement", "label"]
for df in other_datasets:
    df.columns = ["statement", "label"]

cities_datasets = pd.concat(cities_datasets)
other_datasets = pd.concat(other_datasets)
cities_datasets.drop_duplicates(inplace=True)
other_datasets.drop_duplicates(inplace=True)
cities_datasets["is_city"] = True
other_datasets["is_city"] = False

data = pd.concat([cities_datasets.sample(n=6000, random_state=123456), other_datasets.sample(n=6000, random_state=123456)]).reset_index(drop=True)

template = """\
=== BEGIN QUESTION ===
True or False? {question}. Answer must be "True" or "False".
=== END QUESTION ===

=== BEGIN ANSWER ===
{answer}"""

messages = []
for _, row in data.iterrows():
    label = row["label"] == 1
    if not row["is_city"]: label = not label
    messages.append(template.format(question=row["statement"], answer="True" if label else "False"))
data["messages"] = messages

test = pd.concat([data[data["is_city"]].sample(n=1000, random_state=123456), data[~data["is_city"]].sample(n=1000, random_state=123456)]).sample(frac=1).reset_index(drop=True)
train = data.drop(test.index).sample(frac=1).reset_index(drop=True)

train.to_json("/workspace/two-hop/data/geometry_of_truth/train.jsonl", orient="records", lines=True)
test.to_json("/workspace/two-hop/data/geometry_of_truth/test.jsonl", orient="records", lines=True)