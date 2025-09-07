from dataclasses import dataclass
from openai import OpenAI
from pydantic import BaseModel
import yaml


@dataclass
class LLMOutput:
    response: BaseModel
    raw_output: str
    cost: float
    success: bool = True


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

    response = None
    for _ in range(max_retrys):
        try:
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=response_model,
                temperature=temperature,
                top_p=top_p
            )
        except BaseException as e:
            print(f"Error querying OpenRouter: {e}")
            continue
    if response is None:
        return LLMOutput(
            response=None,
            raw_output="",
            cost=0.0,
            success=False
        )

    # get pricing
    models = client.models.list()
    price = 0
    for model in models:
        if model.id == model_name:
            price +=  float(model.pricing["prompt"]) * int(response.usage.prompt_tokens)
            price += float(model.pricing["completion"]) * int(response.usage.completion_tokens)
            break

    return LLMOutput(
        response=response.choices[0].message.parsed,
        raw_output=response.choices[0].message.content,
        cost=price
    )

def get_key(file: str, type: str):
    with open(file, "r") as f:
        return yaml.safe_load(f).get(type, "")