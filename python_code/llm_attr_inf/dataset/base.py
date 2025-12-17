from dataclasses import asdict, dataclass, field
from decimal import Decimal
from enum import IntEnum
import json
import os
from typing import Annotated, Literal
import pandas as pd
import numpy as np
from pydantic import Field, StrictInt, create_model

from llm_attr_inf.utils import fill_in_prompt_file, sample_enum_without_replacement, spaced_out_random_floats, spaced_out_random_ints

REASONING_INSTR = "- `reasoning`: the reasoning behind your answer. IMPORTANT: Make sure to not use quotation marks in the reasoning, as it interferes with JSON parsing!"
CERTAINTY_INSTR = "- `certainty_score`: your certainty, as an INTEGER from 1 to 5, for all three guesses combined"
NUM_GOOD_EXAMPLES = 3
NUM_BAD_EXAMPLES = 2


@dataclass
class ProfileText:
    text: str
    metadata: dict = None
    private_metadata: dict = None

    def format_metadata(self):
        return ", ".join(f"{k}: {v}" for k, v in self.metadata.items())

    def format(self, include_metadata: bool=False, index: int=None):
        prefix = f"Text {index + 1}" if index is not None else "Text"
        metadata_str = f" ({self.format_metadata()})" \
            if include_metadata and self.metadata is not None \
                else ""
        return f"[{prefix}{metadata_str}]\n{self.text}"


class AttributeTypes(IntEnum):
    INT = 0
    FLOAT = 1
    ENUM = 2
    INT_ENUM = 3


@dataclass
class Attribute:
    name: str
    description: str
    type: AttributeTypes
    reasonable_range: tuple[float, float] = field(default=(1, 10))
    numeric_bounds: tuple[float, float] | None = field(default=None)
    enum_values: list[str] | None = field(default=None)

    def output_format(self):
        bounds = self.numeric_bounds
        if self.type == AttributeTypes.INT:
            fmt = "an integer"
            if bounds is not None:
                bounds = [int(bounds[0]), int(bounds[1])]
        elif self.type == AttributeTypes.FLOAT:
            fmt = "a floating point number"
            bounds = [round(bounds[0], 3), round(bounds[1], 3)]
        elif self.type == AttributeTypes.ENUM:
            values = f", ".join([f"\"{val}\"" for val in self.enum_values])
            fmt = f"one of: {{{values}}}"
        elif self.type == AttributeTypes.INT_ENUM:
            fmt = f"an integer representing your guess: "
            fmt += ", ".join([f"{i+1} if {v}" for i, v in enumerate(self.enum_values)])
        else:
            raise NotImplementedError(f"Unknown attribute type {self.type}")

        if self.numeric_bounds is not None:
            b0, b1 = int(self.numeric_bounds[0]), int(self.numeric_bounds[1])
            fmt += f" between {b0} and {b1}"
        return fmt
    
    def response_format_field(self):
        if self.numeric_bounds:
            lb, ub = self.numeric_bounds
        else:
            lb, ub = None, None

        if self.type == AttributeTypes.INT:
            if self.numeric_bounds:
                return Annotated[StrictInt, Field(ge=lb, le=ub)]
            return int
        if self.type == AttributeTypes.FLOAT:
            if self.numeric_bounds:
                return Annotated[Decimal, Field(ge=0.0, le=1.0)]
            return float
        if self.type == AttributeTypes.ENUM:
            return Literal(tuple(self.enum_values))
        if self.type == AttributeTypes.INT_ENUM:
            return Annotated[StrictInt, Field(ge=1, le=len(self.enum_values))]
        raise NotImplementedError(f"Unknown attribute type {self.type}") 

    def random_valid(self, num_samples=1):
        """
        Generates num_samples random valid values for this attribute.
        Makes sure to cover a broad range of reasonable values.
        """

        if self.type == AttributeTypes.INT:
            return spaced_out_random_ints(
                *self.reasonable_range, num_samples
            )
        
        if self.type == AttributeTypes.FLOAT:
            return spaced_out_random_floats(
                *self.reasonable_range, num_samples
            )
        
        if self.type == AttributeTypes.ENUM:
            return sample_enum_without_replacement(
                self.enum_values, num_samples
            )
        
        if self.type == AttributeTypes.INT_ENUM:
            return sample_enum_without_replacement(
                list(range(1, len(self.enum_values) + 1)),
                num_samples
            )
        raise NotImplementedError(f"Unknown attribute type {self.type}")


@dataclass
class Dataset:
    attribute_df: pd.DataFrame
    texts: list[list[ProfileText]]
    texts_description: str
    fields_to_infer: list[Attribute]

    @property
    def num_profiles(self):
        return len(self.texts)
    
    def format_comments(self, index: int, include_metadata=False) -> str:
        single_comment = len(self.texts[index]) == 0
        return "\n\n".join(
            text.format(include_metadata, idx if not single_comment else None) \
                for idx, text in enumerate(self.texts[index])
        )
    
    def get_response_model(self, include_reasoning=False):
        model_kwargs = {
            f"{attr.name}_guess": (attr.response_format_field(), ...) \
                for attr in self.fields_to_infer
        }
        model_kwargs["confidence"] = (
            Annotated[StrictInt, Field(ge=1, le=5)], ...
        )
        if include_reasoning:
            model_kwargs["reasoning"] = (str, ...)
        
        GuessResponse = create_model(
            "GuessResponse", **model_kwargs
        )
        return GuessResponse
    
    def _example_llm_outputs(
        self, reasoning: bool, num_samples=1,
        incomplete=False, bad_certainty=False
    ):
        examples = []
        attr_values = {
            attr.name: attr.random_valid(num_samples) \
                for attr in self.fields_to_infer
        }
        certainties = sample_enum_without_replacement(
            list(range(1, 6)), num_samples
        )
        if bad_certainty:
            certainties = sample_enum_without_replacement(
                [x + 0.5 for x in range(1, 5)], num_samples
            )
        
        if incomplete and len(self.fields_to_infer) == 1:
            fields = [
                f"{self.fields_to_infer[0].name}_guess",
                "certainty_score"
            ]
            # to make an incomplete example, we can't just sample a random
            # attribute because we only have one. So we have to drop one of the
            # certainty or the guess.
            to_drop = sample_enum_without_replacement(fields, num_samples)
        else:
            to_drop = None

        for i in range(num_samples):
            example = {}
            if reasoning:
                example["reasoning"] = "<your reasoning>"
            attrs = [np.random.choice(self.fields_to_infer)] \
                if incomplete else self.fields_to_infer
            for attr in attrs:
                example[f"{attr.name}_guess"] = attr_values[attr.name][i]
            
            example["certainty_score"] =  certainties[i]
            
            if incomplete and len(self.fields_to_infer) == 1:
                del example[to_drop[i]]
            examples.append(json.dumps(example))
        return examples
        
    
    def fill_in_prompt(
        self, index: int,
        prompt_filename: str,
        include_metadata=False,
        reasoning=False,
        random_state: int=42
    ):
        # TODO: use a thread-safe RNG here.
        np.random.seed(random_state)

        instructions_block="\n".join(
            f"- `{x.name}_guess`: {x.output_format()}" for x in self.fields_to_infer
        ) + f"\n{CERTAINTY_INSTR}"
        if reasoning:
            instructions_block = f"{REASONING_INSTR}\n{instructions_block}"
        
        good_examples = self._example_llm_outputs(
            reasoning, num_samples=NUM_GOOD_EXAMPLES + 1
        )
        incomplete_examples = self._example_llm_outputs(
            reasoning, incomplete=True, num_samples=NUM_BAD_EXAMPLES
        )

        return fill_in_prompt_file(
            prompt_filename,
            dict(
                texts_description=self.texts_description,
                attributes=", ".join([x.name  for x in self.fields_to_infer]),
                texts=self.format_comments(index, include_metadata=include_metadata),
                attributes_block="\n".join([
                    f"- {x.name}: {x.description}" for x in self.fields_to_infer
                ]),
                instructions_block=instructions_block,
                good_examples="\n".join([
                    f"Good example {i+1}: {out}" for i, out in enumerate(good_examples[:-1])
                ]),
                bad_certainty_example=self._example_llm_outputs(
                    reasoning, num_samples=1, bad_certainty=True
                )[0],
                incomplete_examples="\n".join([
                    f"WRONG EXAMPLE {i+1}: {out}" \
                    for i, out in enumerate(incomplete_examples)
                ]),
                another_good_example=good_examples[-1],
            )
        )

    def save(self, output_directory: str):
        texts_df = self.attribute_df.copy()
        max_texts = max([len(x) for x in self.texts])
        for i in range(max_texts):
            texts_df[f"text_{i+1}"] = [
                (x[i].text if len(x) > i else None) for x in self.texts
            ]
        os.makedirs(output_directory, exist_ok=True)
        texts_df.to_csv(
            f"{output_directory}/texts.tsv",
            sep="\t",
            index=False
        )

        full_df = self.attribute_df.copy()
        full_df["texts"] = [[asdict(text) for text in x] for x in self.texts]
        full_df.to_json(
            f"{output_directory}/dataset.jsonl",
            lines=True, orient="records"
        )

        with open(f"{output_directory}/metadata.json", "w") as f:
            json.dump({
                "fields_to_infer": [asdict(x) for x in self.fields_to_infer],
                "texts_description": self.texts_description
            }, f, indent=2)
    
    @classmethod
    def load(cls, directory: str):
        df = pd.read_json(
            f"{directory}/dataset.jsonl",
            lines=True, orient="records"
        )
        texts = [
            [ProfileText(**text) for text in x] \
                for x in df["texts"]
        ]
        df.drop("texts", axis=1, inplace=True)
        with open(f"{directory}/metadata.json", "r") as f:
            metadata = json.load(f)
        
        return cls(
            attribute_df=df,
            texts=texts,
            texts_description=metadata["texts_description"],
            fields_to_infer = [
                Attribute(**x) for x in metadata["fields_to_infer"]
            ]
        )

