import json
import os
import jsonlines
from pydantic import BaseModel
import argparse
import multiprocessing

from utils import query_openrouter, LLMOutput, get_key


class GuessResponse(BaseModel):
    income_category_guess: int
    sex_guess: str
    age_guess: int
    certainty_score: float # they should be outputting an integer but some models mess up


class GuessResponseReasoning(BaseModel):
    reasoning: str
    income_category_guess: int
    sex_guess: str
    age_guess: int
    certainty_score: float 


def run_one_line(
    line: dict,
    prompt_file: str,
    system_prompt_file: str,
    model_name: str,
    temperature: float,
    top_p: float,
    include_reasoning: bool
) -> LLMOutput:
    with open(system_prompt_file, "r") as f:
        system_prompt = f.read()
    with open(prompt_file, "r") as f:
        unfilled_prompt = f.read()
    
    comments = "\n\n".join(
        [f"[Text {k+1}] {text}" for k, text in enumerate(line["text"])]
    )
    # look for {{comments}} in the prompt file
    user_prompt = unfilled_prompt.replace("{{comments}}", comments)

    api_key = get_key("keys.yaml", "openrouter")
    llm_output: LLMOutput = query_openrouter(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model_name,
        response_model=GuessResponse if not include_reasoning else GuessResponseReasoning,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p
    )

    print(f"Author: {line['author']}, Model: {model_name}, Temp: {temperature}, Top-p: {top_p}, Cost: {llm_output.cost:.5f} USD")

    return llm_output


def run_for_one_configuration(
    jsonl_input_file: str,
    output_dir: str,
    prompt_file: str,
    system_prompt_file: str,
    model_name: str,
    temperature: float,
    top_p: float,
    max_threads: int = 1,
    include_reasoning: bool = False
) -> float:
    # maps author to response
    output_guesses: list[dict] = []
    total_cost = 0

    inputs = []

    with jsonlines.open(jsonl_input_file) as reader:
        for obj in reader:
            inputs.append(obj)
    
    with multiprocessing.Pool(processes=max_threads) as pool:
        results = pool.starmap(
            run_one_line,
            [
                (
                    obj,
                    prompt_file,
                    system_prompt_file,
                    model_name,
                    temperature,
                    top_p,
                    include_reasoning
                ) for obj in inputs
            ]
        )
    for obj, llm_output in zip(inputs, results):
        total_cost += llm_output.cost
        output_guesses.append({
            "author": obj["author"],
            "success": llm_output.success,
            "llm_output": json.loads(llm_output.response.model_dump_json()) if llm_output.success else None,
            "author_data": obj,
            "llm_config": {
                "model_name": model_name,
                "temperature": temperature,
                "top_p": top_p,
                "cost": llm_output.cost
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
    parser.add_argument("--jsonl_input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--system_prompt_file", type=str, required=True)
    parser.add_argument("--model_names", nargs='+', type=str, default=["openai/o3"])
    parser.add_argument("--temperatures", nargs='+', type=float, default=[0.7])
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_threads", type=int, default=5)
    parser.add_argument("--include_reasoning", action="store_true", help="Whether to include reasoning in the output")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials to run for each configuration")
    args = parser.parse_args()


    total_overall_cost = 0.0
    for trial in range(args.num_trials):
        print(f"Starting trial {trial+1}/{args.num_trials}")
        output_dir = f"{args.output_dir}/trial_{trial+1}"
        os.makedirs(output_dir, exist_ok=True)
        for model_name in args.model_names:
            for temperature in args.temperatures:
                total_overall_cost += run_for_one_configuration(
                    jsonl_input_file=args.jsonl_input_file,
                    output_dir=output_dir,
                    prompt_file=args.prompt_file,
                    system_prompt_file=args.system_prompt_file,
                    model_name=model_name,
                    temperature=temperature,
                    top_p=args.top_p,
                    max_threads=args.max_threads,
                    include_reasoning=args.include_reasoning
                )
    
    print(f"\n\nTotal overall cost for all trials: {total_overall_cost:.5f} USD")