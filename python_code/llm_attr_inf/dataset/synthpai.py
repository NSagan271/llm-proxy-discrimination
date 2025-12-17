from datetime import datetime
import os
import re
import subprocess
import pandas as pd
import jsonlines
import argparse
import itertools
import numpy as np

from llm_attr_inf.dataset.base import ProfileText, Dataset
from llm_attr_inf.dataset.attributes import AGE, AGE_ADULT, GENDER, INCOME_CATEGORY


DATA_URL = "https://github.com/eth-sri/SynthPAI/raw/refs/heads/main/data/synthpai.jsonl"


def load_synthpai(data_dir: str):
    if not os.path.exists(f"{data_dir}/synthpai.jsonl"):
        os.makedirs(data_dir, exist_ok=True)
        subprocess.run(["wget", DATA_URL], cwd=os.path.abspath(data_dir))

    data = []
    with jsonlines.open(f"{data_dir}/synthpai.jsonl") as reader:
        for obj in reader:
            data.append(
                {
                    "author": obj["author"],
                    "username": obj["username"],
                    "age": obj["profile"]["age"],
                    "sex": obj["profile"]["sex"],
                    "city_country": obj["profile"]["city_country"],
                    "birth_city_country": obj["profile"]["birth_city_country"],
                    "income_category": obj["profile"]["income_level"],
                    "text": obj["text"]
                }
            )
    
    df = pd.DataFrame(data)
    df["unmerged_income_category"] = df["income_category"]
    df["income_category"] = df["income_category"].replace(
        {"very high": "high"}
    )

    # 18–34, 35–54, 55+:
    df["age_range"] = pd.cut(
        df["age"], bins=[17, 34, 54, 100], labels=["18-34", "35-54", "55+"]
    )
    return df


def build_synthpai_dataset(
    data_dir: str="data/synthpai",
    output_dir: str="outputs/data/synthpai",
    all_samples: bool = False,
    min_num_samples: int=10,
    max_texts_per_person: int=3,
    seed: int=42
):
    df = load_synthpai(data_dir)

    income_levels = df["income_category"].unique()
    genders = df["sex"].unique()
    age_ranges = df["age_range"].unique()

    # Determine 
    boxes = {}
    for (income, gender, age) in itertools.product(income_levels, genders, age_ranges):
        box = df[(df["income_category"] == income) & (df["sex"] == gender) & (df["age_range"] == age)]["author"].unique()
        if len(box):
            boxes[(income, gender, age)] = box

    samples_per_box = (min_num_samples + len(boxes) - 1) // len(boxes)
    if not all_samples:
        print(len(boxes), "boxes found. Sampling", samples_per_box, "per box.")
    else:
        print(len(boxes), "boxes found. Using all samples in each box.")

    texts = []
    data = []
    np.random.seed(seed)

    for (income, gender, _), box in boxes.items():
        if not all_samples:
            samples = np.random.choice(box, size=min(samples_per_box, len(box)), replace=False)
        else:
            samples = box

        for sample in samples:
            subdf = df[df["author"] == sample]
            row = subdf.iloc[0].to_dict()
            del row["text"]

            sample_texts = []
            profile_texts = subdf["text"].tolist()
            for text in np.random.choice(profile_texts, min(max_texts_per_person, len(profile_texts)), replace=False):
                # some comments are in the form Question: ...\nQuestion description: ...
                # For those, only keep the part after "Question description:"
                match = re.search(r"Question description:\s*(.*)", text)
                if match:
                    sample_texts.append(ProfileText(match.group(1)))
                else:
                    sample_texts.append(ProfileText(text))
            texts.append(sample_texts)
            data.append(row)
        
    dataset = Dataset(
        attribute_df=pd.DataFrame(data),
        texts=texts,
        texts_description="public comments made on Reddit",
        fields_to_infer=[AGE_ADULT, GENDER, INCOME_CATEGORY]
    )
    print(f"Saving dataset to directory {output_dir}")
    dataset.save(output_dir)