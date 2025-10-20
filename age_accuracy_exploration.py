import os
import pandas as pd
import re
import jsonlines
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import warnings
warnings.filterwarnings("ignore")


OUTPUT_DIR = "data/plots"
PROFILE_FILE = 'outputs/data/2025_09_24_19h53m08s/machine_readable/synthpai_samples_trial_1.jsonl'
LLM_CSV = 'outputs/llm_human_experiment/llm_human_experiment_results_full.csv'
HUMAN_SURVEY_CSV = 'data/human_results.csv'
GROUND_TRUTH_CSV = 'outputs/data/ground_truth.csv'

INDIVIDUAL_LLMS = [
    "openai_o3",
    "x-ai_grok-4"
]
# INDIVIDUAL_LLMS = []


# --- Global Style Settings (R-like aesthetic) ---
TITLE_SIZE = 24
LABEL_SIZE = 20
TICK_SIZE = 16
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',          # clean sans-serif similar to ggplot
    'font.size': 18,                       # base size
    'axes.titlesize': TITLE_SIZE,                  # title size
    'axes.labelsize': LABEL_SIZE,                  # axis label size
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'axes.spines.top': False,              # remove top/right spines (ggplot look)
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'axes.grid': False,                     # subtle grid
    'grid.color': '#dddddd',               # light gray grid
    'grid.linestyle': '-',                 
    'grid.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})


def get_ground_truth():
    ground_truth = pd.read_csv(GROUND_TRUTH_CSV)
    ground_truth["true_age_bin"] = (ground_truth["true_age"] >= 40).astype(int)
    return ground_truth


def get_human_dataframes() -> dict[str, pd.DataFrame]:
    # get profile order
    profile_order = []
    with jsonlines.open(PROFILE_FILE, 'r') as reader:
        for obj in reader:
            profile_order.append(obj['author'])
    
    # Same cleaning code as in R code
    df = pd.read_csv(HUMAN_SURVEY_CSV)
    df = df[(df["Q1.2"] == "I agree to participate") & \
        (df["Q2.1"] != "Under 18") & \
        (df["Q2.2"] != "I do not reside in the United States") & \
        (df["Q3.1"] == "Somewhat disagree")]
    
    col_regex = re.compile(r"^\d+_(Q7\.3_1|Q7\.4|Q7\.5|Q7\.7|Q7\.6_Page Submit)$")
    ans_cols = [
        col for col in df.columns if col_regex.match(col)
    ]
    wanted_columns = [
        "ResponseId", "treatment"
    ] + ans_cols
    df = df[wanted_columns]

    new_df = pd.DataFrame(columns=[
        "ResponseId", "treatment", "profile", "Q7.3_1", "Q7.4", "Q7.5", "Q7.7", "Q7.6_Page Submit"
    ])
    for i in range(300): # number of profiles
        col_rename = {
            f"{i+1}_{col}": col for col in ["Q7.3_1", "Q7.4", "Q7.5", "Q7.7", "Q7.6_Page Submit"]
        }
        temp_df = df[["ResponseId", "treatment"] + list(col_rename.keys())].copy().rename(columns=col_rename)
        temp_df = temp_df[temp_df["Q7.6_Page Submit"].notna()]
        temp_df["profile"] = profile_order[i]
        new_df = pd.concat([new_df, temp_df], ignore_index=True)

    new_df = new_df.rename(columns={
        "Q7.3_1": "age_guess",
        "Q7.4": "sex_guess",
        "Q7.5": "income_guess",
        "Q7.7": "certainty",
        "Q7.6_Page Submit": "inferencepage_timer"
    })

    # filter out bad guesses
    new_df = new_df[new_df["age_guess"].between(18, 100)]

    ground_truth = get_ground_truth()

    new_df = new_df[new_df["age_guess"].between(18,100)].merge(
        ground_truth[['profile', 'true_age', 'true_age_bin']], on='profile'
    )
    new_df['age_bin_guess'] = (new_df['age_guess'] >= 40).astype(int)

    return {
        "control":  new_df[new_df["treatment"] == "control"],
        "timepressure":  new_df[new_df["treatment"] == "timepressure"]
    }


def get_llm_dataframe():
    llm = pd.read_csv(LLM_CSV)
    # filter out bad guesses
    llm = llm[llm['age_guess'].between(18,100)]
    llm['age_bin_guess'] = (llm['age_guess'] >= 40).astype(int)
    llm['true_age_bin'] = (llm['true_age'] >= 40).astype(int)
    return llm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    human = get_human_dataframes()["control"]
    llm = get_llm_dataframe()
    ground_truth = get_ground_truth()

    TO_ANALYZE = {
        "Human": human,
        "LLM": llm,
    }
    for llm_name in INDIVIDUAL_LLMS:
        TO_ANALYZE[llm_name] = llm[llm['arm'] == f"{llm_name}_temp_0.0_top_p_1.0"]
    

    COLORS = [
        '#377eb8', # blue
        '#ff7f00', # orange
        '#4daf4a', # green
        '#e41a1c', # red
        '#984ea3', # purple
        '#ffff33', # yellow
        '#a65628', # brown
        '#f781bf', # pink
    ]
    COLORS_PER_KEY = {key: COLORS[i % len(COLORS)] for i, key in enumerate(TO_ANALYZE.keys())}

    ###########################################################################
    ###########################################################################
    # Age guess error distribution by model
    age_diffs = {
        key: value.assign(
            age_diff=lambda x: x['age_guess'] - x['true_age']
        )["age_diff"]
        for key, value in TO_ANALYZE.items()
    }
    
    plt.figure(figsize=(14, 8))
    for label, age_diff in age_diffs.items():
        density, bins, _ = plt.hist(age_diff, bins=30, density=True, alpha=0.5, label=label, color=COLORS_PER_KEY[label])
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        plt.plot(bin_centers, density, marker='o', color=COLORS_PER_KEY[label])
    plt.title('Age Guess Error Distribution (Guess - True Age)')
    plt.xlabel('Age Guess Error')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/age_guess_error_distribution.png")

    print("=" * 40)
    for key, age_diff in age_diffs.items():
        print(f"Age Guess Error for {key}: mean={float(np.mean(age_diff)):.3f}, std={float(np.std(age_diff)):.3f}, median={float(np.median(age_diff)):.3f}")
    print("=" * 40)
    print()
    ###########################################################################

    ###########################################################################
    # Age guess distribution by model
    plt.figure(figsize=(14, 8))
    plt.grid(True)
    density, bins, _ = plt.hist(ground_truth['true_age'], bins=30, density=True, label="True Ages", alpha=0.5, color='#984ea3')
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    plt.plot(bin_centers, density, marker='o', color='#984ea3')

    for label, df_ in TO_ANALYZE.items():
        density, bins, _ = plt.hist(df_['age_guess'], bins=30, density=True, alpha=0.5, label=label, color=COLORS_PER_KEY[label])
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        plt.plot(bin_centers, density, marker='o', color=COLORS_PER_KEY[label])
    plt.title('Age Guess Distribution')
    plt.xlabel('Age Guess')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/age_guess_distribution.png")

    print("=" * 40)
    for key, df_ in TO_ANALYZE.items():
        age_guess = df_['age_guess']
        mean, std, med = float(np.mean(age_guess)), float(np.std(age_guess)), float(np.median(age_guess))
        print(f"Age Guess for {key}: mean={mean:.3f}, std={std:.3f}, median={med:.3f}")
    print("-----")
    true_age = ground_truth['true_age']
    true_mean, true_std, true_med = float(np.mean(true_age)), float(np.std(true_age)), float(np.median(true_age))
    print(f"True Age: mean={true_mean:.3f}, std={true_std:.3f}, median={true_med:.3f}")
    print("=" * 40)
    print()
    ###########################################################################

    ###########################################################################
    # Exploratory plots
    age_bins = list(range(18, 101, 3))
    filled_age_bins = []
    binned_data = {}

    first = True
    for key, df_ in TO_ANALYZE.items():
        binned_data[key] = []
        for i in range(len(age_bins)-1):
            bin_start = age_bins[i]
            bin_end = age_bins[i+1]
            profiles = set(ground_truth[(ground_truth['true_age'] >= bin_start) & (ground_truth['true_age'] < bin_end)]['profile'])
            if len(profiles) == 0:
                continue
            df_bin = df_[df_['profile'].isin(profiles)].copy()
            if len(df_bin) == 0:
                continue

            if first:
                filled_age_bins.append(bin_start)
            binned_data[key].append({
                'model': key,
                'age_bin_start': bin_start,
                'age_bin_end': bin_end,
                'num_profiles': len(profiles),
                'num_samples': len(df_bin),
                'binned_error_rate': np.mean(df_bin['age_bin_guess'] != df_bin['true_age_bin']),
                'pm5_error_rate': np.mean(np.abs(df_bin['age_guess'] - df_bin['true_age']) > 5),
                'avg_age_diff': np.mean(df_bin['age_guess'] - df_bin['true_age']),
                'avg_abs_age_diff': np.mean(np.abs(df_bin['age_guess'] - df_bin['true_age'])),
            })
        first = False

    binned_dfs = {key: pd.DataFrame(data) for key, data in binned_data.items()}
    plt.figure(figsize=(14, 7))

    for (key, binned_df) in binned_dfs.items():
        binned_error_rate_per_bin = binned_df['binned_error_rate']
        plt.plot(filled_age_bins, binned_error_rate_per_bin, marker='o', linestyle='-', linewidth=2, label=key, color=COLORS_PER_KEY[key])
    plt.xlabel('True Age Bin Start')
    plt.ylabel('Age Bin Guess Error Rate')
    plt.title('Age Bin Guess Error Rate by True Age')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/age_bin_error_rate_by_true_age.png")

    plt.figure(figsize=(14, 7))
    for (key, binned_df) in binned_dfs.items():
        pm5_error_rate_per_bin = binned_df['pm5_error_rate']
        plt.plot(filled_age_bins, pm5_error_rate_per_bin, marker='o', linestyle='-', linewidth=2, label=key, color=COLORS_PER_KEY[key])
    plt.xlabel('True Age Bin Start')
    plt.ylabel('Age +/- 5 Guess Error Rate')
    plt.title('Age +/- 5 Guess Error Rate by True Age')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/age_pm5_error_rate_by_true_age.png")

    plt.figure(figsize=(14, 7))
    for (key, binned_df) in binned_dfs.items():
        avg_abs_age_diff_per_bin = binned_df['avg_abs_age_diff']
        plt.plot(filled_age_bins, avg_abs_age_diff_per_bin, marker='o', linestyle='-', linewidth=2, label=key, color=COLORS_PER_KEY[key])
    plt.xlabel('True Age Bin Start')
    plt.ylabel('Average Absolute Age Difference')
    plt.title('Average Absolute Age Difference |Guess - True| by True Age')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/age_error_by_true_age.png")
    ###########################################################################

    ###########################################################################
    # +/- 5 correct but bin wrong

    bin_right = {
        key: df_["true_age_bin"] == df_["age_bin_guess"] 
        for key, df_ in TO_ANALYZE.items()
    }
    bin_wrong = {
        key: df_["true_age_bin"] != df_["age_bin_guess"] 
        for key, df_ in TO_ANALYZE.items()
    }
    pm5_correct = {
        key: (df_["age_guess"] - df_["true_age"]).abs() <= 5
        for key, df_ in TO_ANALYZE.items()
    }


    cross_proportion = {
        key: len(df_[bin_wrong[key] & pm5_correct[key]]) / len(df_)
        for key, df_ in TO_ANALYZE.items()
    }
    non_cross_proportion = {
        key: len(df_[bin_right[key] & pm5_correct[key]]) / len(df_)
        for key, df_ in TO_ANALYZE.items()
    }

    print("="*40)
    for key in TO_ANALYZE:
        print(f"{key} +/-5 accuracy: {100*(cross_proportion[key] + non_cross_proportion[key]):.3f}%")
    print()
    for key in TO_ANALYZE:
        print(f"{key}: {100*cross_proportion[key]:.3f}% had bin wrong, +/-5 correct; {100*non_cross_proportion[key]:.3f}% had bin and +/-5 correct")
    print("="*40)
    print()

    fig, axes = plt.subplots(1, 2, figsize=(24 * 0.9, 6))

    axes[0].bar(cross_proportion.keys(),
                cross_proportion.values(),
                color=[COLORS_PER_KEY[key] for key in cross_proportion.keys()])
    axes[0].set_ylabel('Proportion', fontsize=LABEL_SIZE)
    axes[0].set_title('Proportion of Guesses with +/- 5 Age Correct, Bin Wrong', fontsize=TITLE_SIZE)
    axes[0].tick_params(axis='x', labelsize=LABEL_SIZE)
    axes[0].tick_params(axis='y', labelsize=TICK_SIZE)

    # --- Plot 2 ---
    axes[1].bar(non_cross_proportion.keys(),
                non_cross_proportion.values(),
                color=[COLORS_PER_KEY[key] for key in non_cross_proportion.keys()])
    axes[1].set_ylabel('Proportion', fontsize=LABEL_SIZE)
    axes[1].set_title('Proportion of Guesses with +/- 5 Age and Bin Correct', fontsize=TITLE_SIZE)
    axes[1].tick_params(axis='x', labelsize=LABEL_SIZE)
    axes[1].tick_params(axis='y', labelsize=TICK_SIZE)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pm5_and_bin_error_proportions.png")
    ###########################################################################

    ###########################################################################
    # Guesses in each bin
    prop_of_guesses_over_40 = {
        key: len(df_[df_["age_bin_guess"] == 1]) / len(df_) \
            for key, df_ in TO_ANALYZE.items()
    }
    pm5_accuracy_with_true_age_under_40 = {
        key: len(df_[(df_["true_age_bin"] == 0) & (np.abs(df_["age_guess"] - df_["true_age"]) <= 5)]) \
            / len(df_[df_["true_age_bin"] == 0])
        for key, df_ in TO_ANALYZE.items()
    }

    pm5_accuracy_with_true_age_over_40 = {
        key: len(df_[(df_["true_age_bin"] == 1) & (np.abs(df_["age_guess"] - df_["true_age"]) <= 5)]) \
            / len(df_[df_["true_age_bin"] == 1])
        for key, df_ in TO_ANALYZE.items()
    }

    print("="*40)
    for key in TO_ANALYZE:
        print(f"{key}: {100*prop_of_guesses_over_40[key]:.3f}% of guesses were over 40.")
        print(f"{key}: For guesses >= 40, +/5 accuracy was {100*pm5_accuracy_with_true_age_over_40[key]:.3f}%.")
        print(f"{key}: For guesses <40, +/5 accuracy was {100*pm5_accuracy_with_true_age_under_40[key]:.3f}%.")
        print()
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    axes[0].bar(prop_of_guesses_over_40.keys(),
                prop_of_guesses_over_40.values(),
                color=[COLORS_PER_KEY[key] for key in prop_of_guesses_over_40.keys()])
    axes[0].set_ylabel('Proportion', fontsize=LABEL_SIZE)
    axes[0].set_title('Proportion of Guesses Over 40', fontsize=TITLE_SIZE)
    axes[0].tick_params(axis='x', labelsize=LABEL_SIZE)
    axes[0].tick_params(axis='y', labelsize=TICK_SIZE)

    # --- Plot 2 ---
    axes[1].bar(pm5_accuracy_with_true_age_under_40.keys(),
                pm5_accuracy_with_true_age_under_40.values(),
                color=[COLORS_PER_KEY[key] for key in pm5_accuracy_with_true_age_under_40.keys()])
    axes[1].set_ylabel('Proportion', fontsize=LABEL_SIZE)
    axes[1].set_title('Accuracy for Profiles Under 40 (+/-5)', fontsize=TITLE_SIZE)
    axes[1].tick_params(axis='x', labelsize=LABEL_SIZE)
    axes[1].tick_params(axis='y', labelsize=TICK_SIZE)

    # --- Plot 3 ---
    axes[2].bar(pm5_accuracy_with_true_age_over_40.keys(),
                pm5_accuracy_with_true_age_over_40.values(),
                color=[COLORS_PER_KEY[key] for key in pm5_accuracy_with_true_age_over_40.keys()])
    axes[2].set_ylabel('Proportion', fontsize=LABEL_SIZE)
    axes[2].set_title('Accuracy for Profiles Over 40 (+/-5)', fontsize=TITLE_SIZE)
    axes[2].tick_params(axis='x', labelsize=LABEL_SIZE)
    axes[2].tick_params(axis='y', labelsize=TICK_SIZE)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/guesses_over_40.png")


if __name__ == "__main__":
    main()
