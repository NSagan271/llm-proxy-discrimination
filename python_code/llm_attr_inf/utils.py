import re

import numpy as np


def fill_in_prompt(
    template: str,
    arguments: dict[str, str]
):
    for key, value in arguments.items():
        template = template.replace("{{" + key + "}}", value)
    
    # Check for any unreplaced placeholders
    unreplaced = re.findall(r"{{(.*?)}}", template)
    if unreplaced:
        print(f"[WARNING] Unreplaced placeholders in template: {unreplaced}")
    return template


def load_prompt_file(template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def fill_in_prompt_file(
    template_path: str,
    arguments: dict[str, str]
):
    template = load_prompt_file(template_path)
    return fill_in_prompt(template, arguments)


def spaced_out_random_ints(
    min: int,
    max: int,
    num_samples=1,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    bins = np.linspace(
        min, max,
        num_samples + 1
    )
    bins = [int(round(b)) for b in bins]
    samples = [
        int(rng.integers(bins[i], bins[i+1])) \
            for i in range(num_samples)
    ]
    rng.shuffle(samples)
    return samples


def spaced_out_random_floats(
    min: int,
    max: int,
    num_samples=1,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()
    bins = np.linspace(
        min, max,
        num_samples + 1
    )
    samples = [
        float(rng.uniform(bins[i], bins[i+1])) \
            for i in range(num_samples)
    ]
    rng.shuffle(samples)
    return samples


def sample_enum_without_replacement(
    choices: list,
    num_samples=1,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()
    enum_values = choices[:]
    while len(enum_values) < num_samples:
        enum_values = enum_values + choices
    rng.shuffle(enum_values)
    return enum_values[:num_samples]
