from datetime import datetime
import os
import subprocess
import pandas as pd
import jsonlines
import argparse
import itertools


DATA_URL = "https://github.com/eth-sri/SynthPAI/raw/refs/heads/main/data/synthpai.jsonl"


def main(
    data_dir: str="data",
    output_dir: str="outputs",
    min_num_samples: int=15,
    num_trials: int = 10
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
    df = df[df["city_country"].str.match(".*(United States|USA).*", case=True)]

    income_levels = df["income_level"].unique()
    genders = df["sex"].unique()

    boxes = {}
    for (income, gender) in itertools.product(income_levels, genders):
        box = df[(df["income_level"] == income) & (df["sex"] == gender)]
        if len(box):
            boxes[(income, gender)] = box
    
    samples_per_box = (min_num_samples + len(boxes) - 1) // len(boxes)

    time = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
    output_dir += f"/{time}"
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_trials):
        samples = []
        for (income, gender), box in boxes.items():
            sampled = box.sample(n=min(samples_per_box, len(box)), replace=False, random_state=i)
            samples.append(sampled)
        result_df = pd.concat(samples).reset_index(drop=True)
        output_path = os.path.join(output_dir, f"synthpai_samples_trial_{i+1}.csv")
        result_df.to_csv(output_path, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SynthPAI dataset.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to store data files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to store outputs")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of samples to process")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of times to generate samples")
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_num_samples=args.num_samples,
        num_trials=args.num_trials
    )