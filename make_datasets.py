from datetime import datetime
import os
import subprocess
import pandas as pd
import jsonlines
import argparse
import itertools
import numpy as np


DATA_URL = "https://github.com/eth-sri/SynthPAI/raw/refs/heads/main/data/synthpai.jsonl"


def main(
    data_dir: str="data",
    output_dir: str="outputs",
    min_num_samples: int=10,
    num_trials: int = 10,
    max_texts_per_person: int = 10,
    seed: int=42
):
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
                    "income_level": obj["profile"]["income_level"],
                    "text": obj["text"]
                }
            )
    
    df = pd.DataFrame(data)
    df["merged_income_level"] = df["income_level"].replace(
        {"very high": "high"}
    )

    # 18–34, 35–54, 55+:
    df["age_range"] = pd.cut(
        df["age"], bins=[17, 34, 54, 100], labels=["18-34", "35-54", "55+"]
    )

    income_levels = df["merged_income_level"].unique()
    genders = df["sex"].unique()
    age_ranges = df["age_range"].unique()

    boxes = {}
    for (income, gender, age) in itertools.product(income_levels, genders, age_ranges):
        box = df[(df["merged_income_level"] == income) & (df["sex"] == gender) & (df["age_range"] == age)]["author"].unique()
        if len(box):
            boxes[(income, gender, age)] = box

    samples_per_box = (min_num_samples + len(boxes) - 1) // len(boxes)
    print(len(boxes), "boxes found. Sampling", samples_per_box, "per box.")

    time = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
    output_dir += f"/{time}"
    os.makedirs(f"{output_dir}/jsonl", exist_ok=True)
    os.makedirs(f"{output_dir}/human_readable", exist_ok=True)

    for i in range(num_trials):
        trial = []
        for (income, gender, age_range), box in boxes.items():
            np.random.seed(seed + i)
            samples = np.random.choice(box, size=min(samples_per_box, len(box)), replace=False)

            for sample in samples:
                subdf = df[df["author"] == sample]
                row = subdf.iloc[0].to_dict()
                texts = subdf["text"].tolist()
                row["text"] = list(np.random.choice(texts, min(max_texts_per_person, len(texts)), replace=False))
                trial.append(row)

        output_path = os.path.join(output_dir, f"jsonl/synthpai_samples_trial_{i+1}.jsonl")
        with open(output_path, "w") as f:
            with jsonlines.Writer(f) as writer:
                writer.write_all(trial)

        # Write human-readable output for each trial
        human_readable_path = os.path.join(output_dir, f"human_readable/synthpai_samples_trial_{i+1}.txt")
        with open(human_readable_path, "w", encoding="utf-8") as f:
            for row in trial:
                f.write(f"Author: {row['author']} (Username: {row['username']})\n")
                f.write(f"Age: {row['age']}, Sex: {row['sex']}\n")
                f.write(f"City/Country: {row['city_country']}, Birth City/Country: {row['birth_city_country']}\n")
                f.write(f"Income Level: {row['income_level']}\n\n")
                for (k, text) in enumerate(row["text"]):
                    f.write(f"[Text {k+1}] {text}\n\n")
                f.write("\n" + "="*60 + "\n\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SynthPAI dataset.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to store data files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to store outputs")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of samples to process")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of times to generate samples")
    parser.add_argument("--max_texts_per_person", type=int, default=3, help="Maximum number of texts to include per unique person")
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_num_samples=args.num_samples,
        num_trials=args.num_trials,
        max_texts_per_person=args.max_texts_per_person
    )