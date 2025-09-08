from dataclasses import dataclass
import random
import re
import time
from openai import OpenAI
from pydantic import BaseModel
import yaml


@dataclass
class LLMOutput:
    response: BaseModel
    cost: float
    success: bool = True


def compute_cost_openrouter(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    api_key: str
) -> float:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    models = client.models.list()
    price = 0
    for model in models:
        if model.id == model_name:
            price +=  float(model.pricing["prompt"]) * prompt_tokens
            price += float(model.pricing["completion"]) * completion_tokens
            break
    return price


def query_openrouter(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    response_model: BaseModel,
    api_key: str,
    temperature: float=0.7,
    top_p: float=1.0,
    max_retrys: int=3
) -> LLMOutput:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    parsed = None
    total_input_tokens = 0
    total_output_tokens = 0

    for i in range(max_retrys):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                top_p=top_p
            )
            total_input_tokens += int(response.usage.prompt_tokens)
            total_output_tokens += int(response.usage.completion_tokens)

            # try to parse JSON in the output using regex, as not all models support structured output
            matches = re.findall(r"\{.*?\}", response.choices[0].message.content, re.DOTALL)
            for match in matches:
                try:
                    parsed = response_model.model_validate_json(match)
                    break
                except BaseException as e:
                    print(f"Error parsing JSON in match: {e}")
                    continue
            assert parsed is not None, f"Could not parse JSON in the output: {response.choices[0].message.content}"
        except BaseException as e:
            print(f"Error querying OpenRouter: {e}")
            if 'X-RateLimit-Reset' in str(e):
                reset_time = int(re.search(r'X-RateLimit-Reset\'\: \'(\d+)', str(e)).group(1))
                current_time = int(time.time() * 1000)
                wait_time = (reset_time - current_time) / 1000.0 + 1.0
                time.sleep(wait_time * random.uniform(1, 2))
            elif "rate limit" in str(e).lower() or "429" in str(e):
                wait_time = ((i+1)**2 * 15) * 15
                time.sleep(wait_time * random.uniform(1, 2))
            continue
    if parsed is None:
        return LLMOutput(
            response=None,
            cost=compute_cost_openrouter(model_name, total_input_tokens, total_output_tokens, api_key),
            success=False
        )
    print(parsed)


    return LLMOutput(
        response=parsed,
        cost=compute_cost_openrouter(model_name, total_input_tokens, total_output_tokens, api_key),
    )

def get_key(file: str, type: str):
    with open(file, "r") as f:
        return yaml.safe_load(f).get(type, "")