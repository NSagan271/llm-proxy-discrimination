import os
import jsonlines
import argparse
import pandas as pd


INCOME_STRINGS = [
    "invalid",
    "low",
    "middle",
    "high"
]


def main(
    llm_input_folder: str,
    output_folder: str
):
    results = []
    for file in os.listdir(llm_input_folder):
        if not (file.startswith("llm_guesses_") and file.endswith(".jsonl")):
            continue
        model_arm = file[len("llm_guesses_"):-len(".jsonl")]
        with open(os.path.join(llm_input_folder, file), "r") as f:
            for obj in jsonlines.Reader(f):
                # get percent correct
                if "llm_output" in obj:
                    guess = obj["llm_output"]
                    author_data = obj["author_data"]
                    row = {
                        "arm": model_arm, "author": obj["author"], "success": obj["success"],
                        "true_age": author_data["age"],
                        "true_sex": author_data["sex"],
                        "true_income_level": author_data["income_category"].lower()
                    }

                    if obj["success"]:

                        guess["income_level_guess"] = INCOME_STRINGS[guess["income_category_guess"]] \
                            if guess["income_category_guess"] < len(INCOME_STRINGS) else "invalid"
                        del guess["income_category_guess"]
                        row.update(guess)
                    else:
                        row.update({
                            "income_level_guess": "invalid",
                            "age_guess": -999,
                            "sex_guess": ""
                        })
                    results.append(row)
    results = pd.DataFrame(results)
    results["successes_sex_overall"] = (results["true_sex"] == results["sex_guess"]) .astype(int)
    results["successes_income_overall"] = (results["true_income_level"] == results["income_level_guess"]).astype(int)

    # two age tests: (1) binning under 40 vs. 40+, (2) within 5 years
    results["true_age_bin"] = (results["true_age"] < 40).astype(int)
    results["age_guess_bin"] = (results["age_guess"] < 40).astype(int)
    results.loc[results["success"] == False, "age_guess_bin"] = -1  # invalid guess
    results["successes_agebin_overall"] = (results["true_age_bin"] == results["age_guess_bin"]).astype(int)
    results["successes_agein5_overall"] = ((results["true_age"] - results["age_guess"]).abs() <= 5).astype(int)

    # "if answered" columns
    results["successes_agebin_answered"] = results["successes_agebin_overall"].where(results["success"], other=pd.NA)
    results["successes_agein5_answered"] = results["successes_agein5_overall"].where(results["success"], other=pd.NA)
    results["successes_sex_answered"] = results["successes_sex_overall"].where(results["success"], other=pd.NA)
    results["successes_income_answered"] = results["successes_income_overall"].where(results["success"], other=pd.NA)

    # "trial numbers" columns
    results["trials_overall"] = 1
    results["trials_answered_sex"] = results["success"].astype(int)
    results["trials_answered_age"] = results["success"].astype(int)
    results["trials_answered_income"] = results["success"].astype(int)

    # renaming columns
    results = results.rename(columns={
        "author": "profile",
    })
    
    os.makedirs(output_folder, exist_ok=True)
    results.to_csv(os.path.join(output_folder, "llm_human_experiment_results_full.csv"), index=False)

    # also just save the correct/wrong columns
    results[[
        "arm", "profile", "success", "certainty_score",
        "successes_agebin_overall", "successes_agein5_overall",
        "successes_sex_overall", "successes_income_overall",
        "successes_agebin_answered", "successes_agein5_answered",
        "successes_sex_answered", "successes_income_answered",
        "trials_overall", "trials_answered_sex", "trials_answered_age", "trials_answered_income"
    ]].to_csv(os.path.join(output_folder, "llm_human_experiment_results.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="outputs/llm_human_experiment")
    args = parser.parse_args()

    main(
        llm_input_folder=args.llm_input_folder,
        output_folder=args.output_folder
    )

