import pandas as pd
import argparse
import os

def main(
    input_jsonl: str,
    output_dir: str
):
    df = pd.read_json(input_jsonl, lines=True)
    df["profile"] = df["author"]
    df["true_age"] = df["age"]
    df["true_income"] = df["income_category"]
    df["true_sex"] = df["sex"].str.lower()
    df = df[["profile", "true_age", "true_income", "true_sex"]]
    df["author_num"] = df["profile"].str.extract(r'(\d+)$').astype(int)
    df = df.sort_values(by=["author_num"]).drop(columns=["author_num"])

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "ground_truth.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output CSV file")
    args = parser.parse_args()
    main(args.input_jsonl, args.output_dir)

