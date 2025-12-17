from concurrent.futures import ThreadPoolExecutor
import json
import os
import jsonlines
from pydantic import BaseModel
import argparse
import multiprocessing

from llm_attr_inf.experiment.llm import LLMOutput, get_key, query_openrouter
from llm_attr_inf.dataset.base import Dataset


def run_one_line(
    dataset: Dataset,
    index: int,
    prompt_file: str,
    system_prompt_file: str,
    model_name: str,
    temperature: float,
    top_p: float,
    include_reasoning: bool,
    max_retries: int=5
) -> LLMOutput:
    with open(system_prompt_file, "r") as f:
        system_prompt = f.read()
    
    user_prompt = dataset.fill_in_prompt(
        index, prompt_file, reasoning=include_reasoning,
        random_state=index
    )

    api_key = get_key("keys.yaml", "openrouter")
    llm_output: LLMOutput = query_openrouter(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model_name,
        response_model=dataset.get_response_model(include_reasoning),
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_retries=max_retries
    )

    return llm_output


def run_for_one_configuration(
    dataset_dir: str,
    output_dir: str,
    prompt_file: str,
    system_prompt_file: str,
    model_name: str,
    temperature: float,
    top_p: float,
    max_threads: int = 1,
    include_reasoning: bool = False,
    max_retries: int = 5
) -> float:
    # maps author to response
    output_guesses: list[dict] = []
    total_cost = 0

    inputs = []

    dataset = Dataset.load(dataset_dir)
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        results = list(
            executor.map(
                lambda args: run_one_line(*args),
                [
                    (
                        dataset,
                        i,
                        prompt_file,
                        system_prompt_file,
                        model_name,
                        temperature,
                        top_p,
                        include_reasoning,
                        max_retries
                    )
                    for i in range(dataset.num_profiles)
                ]
            )
        )

    for i in range(dataset.num_profiles):
        llm_output = results[i]

        total_cost += llm_output.cost
        output_guesses.append({
            "index": i,
            "success": llm_output.success,
            "llm_output": json.loads(llm_output.response.model_dump_json()) if llm_output.success else None,
            "num_retries": llm_output.num_retries,
            "author_data": dataset.attribute_df.iloc[i].to_dict(),
            "cost": llm_output.cost,
            "llm_config": {
                "model_name": model_name,
                "temperature": temperature,
                "top_p": top_p,
            }
        })

    os.makedirs(output_dir, exist_ok=True)

    cleaned_model_name = model_name.replace("/", "_")
    with jsonlines.open(f"{output_dir}/llm_guesses_{cleaned_model_name}_temp_{temperature}_top_p_{top_p}.jsonl", mode="w") as writer:
        for guess in output_guesses:
            writer.write(guess)
    
    return total_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--system_prompt_file", type=str, required=True)
    parser.add_argument("--model_names_file", type=str, default="model_names.txt", help="File containing model names, one per line")
    parser.add_argument("--temperatures", nargs='+', type=float, default=[0.7])
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_threads", type=int, default=5)
    parser.add_argument("--include_reasoning", action="store_true", help="Whether to include reasoning in the output")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials to run for each configuration")
    parser.add_argument("--max_retries", type=int, default=5, help="Maximum number of retries for LLM queries")
    args = parser.parse_args()


    total_overall_cost = 0.0
    for trial in range(args.num_trials):
        print(f"Starting trial {trial+1}/{args.num_trials}")
        output_dir = f"{args.output_dir}/trial_{trial+1}"
        os.makedirs(output_dir, exist_ok=True)

        with open(args.model_names_file, "r") as f:
            model_names = [line.strip() for line in f if line.strip()]

        for model_name in model_names:
            for temperature in args.temperatures:
                incremental_cost = run_for_one_configuration(
                    dataset_dir=args.dataset_dir,
                    output_dir=output_dir,
                    prompt_file=args.prompt_file,
                    system_prompt_file=args.system_prompt_file,
                    model_name=model_name,
                    temperature=temperature,
                    top_p=args.top_p,
                    max_threads=args.max_threads,
                    include_reasoning=args.include_reasoning,
                    max_retries=args.max_retries
                )
                total_overall_cost += incremental_cost
                print(f"Incremental cost for model {model_name} at temp {temperature}: {incremental_cost:.5f} USD")
    
    print(f"\n\nTotal overall cost for all trials: {total_overall_cost:.5f} USD")