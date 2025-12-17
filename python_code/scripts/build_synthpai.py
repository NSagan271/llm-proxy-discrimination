#!/usr/bin/env python3

import argparse
import os
from llm_attr_inf.dataset.synthpai import build_synthpai_dataset
from llm_attr_inf.dataset.base import Dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build the SynthPAI dataset"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/synthpai",
        help="Path to input SynthPAI data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/data/synthpai",
        help="Path to output directory",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Use all available samples (overrides min_num_samples)",
    )
    parser.add_argument(
        "--min-num-samples",
        type=int,
        default=10,
        help="Minimum number of samples per person",
    )
    parser.add_argument(
        "--max-texts-per-person",
        type=int,
        default=3,
        help="Maximum number of texts per person",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--test-prompt",
        action="store_true",
        help="Test out the dataset with a sample prompt after building",
    )

    args = parser.parse_args()

    build_synthpai_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        all_samples=args.all_samples,
        min_num_samples=args.min_num_samples,
        max_texts_per_person=args.max_texts_per_person,
        seed=args.seed,
    )

    # load in synthpai dataset and test prompt
    sample_prompt = "prompts/staab_et_al_query_partial_attributes.txt"
    if not os.path.exists(sample_prompt) or not args.test_prompt:
        return
    print(f"Found sample prompt; testing out dataset.")
    dataset = Dataset.load(args.output_dir)
    print("\n========\nBasic prompt:\n========")
    print(dataset.fill_in_prompt(
        index=123,
        prompt_filename=sample_prompt,
        include_metadata=False,
        reasoning=False,
        random_state=42
    ))

    print("\n========\nWith metadata:\n========")
    print(dataset.fill_in_prompt(
        index=123,
        prompt_filename=sample_prompt,
        include_metadata=True,
        reasoning=False,
        random_state=43
    ))

    print("\n========\nWith reasoning:\n========")
    print(dataset.fill_in_prompt(
        index=123,
        prompt_filename=sample_prompt,
        include_metadata=True,
        reasoning=True,
        random_state=44
    ))


if __name__ == "__main__":
    main()
