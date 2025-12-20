import argparse
import os

from llm_attr_inf.dataset.blog import build_blog_dataset
from llm_attr_inf.dataset.base import Dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build the blog dataset"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/blogs",
        help="Path to input blog data folder",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/data/blogs",
        help="Path to output directory",
    )
    parser.add_argument(
        "--num-sample",
        type=int,
        default=600,
        help="Number of blogs to sample",
    )
    parser.add_argument(
        "--min-blog-len",
        type=int,
        default=300,
        help="Minimum blog length (characters)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=400,
        help="Truncation length (characters)",
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=350,
        help="Target blog length (characters)",
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

    build_blog_dataset(
        data_folder=args.data_dir,
        output_dir=args.output_dir,
        num_sample=args.num_sample,
        min_blog_len=args.min_blog_len,
        target_length=args.target_len,
        max_length=args.max_len,
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
