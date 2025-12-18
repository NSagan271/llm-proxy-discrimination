"""
Build a balanced blog dataset by gender and age bins.

- Parses XML blog files
- Samples users by demographic bins
- Selects and truncates one blog post per user
- Writes JSONL and TSV outputs
"""

from __future__ import annotations

from glob import glob
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from lxml import etree

from llm_attr_inf.dataset.base import ProfileText, Dataset
from llm_attr_inf.dataset.attributes import AGE, GENDER


SEED = 42
MIN_BLOG_LEN = 300
TRUNC_LEN = 350

FILENAME_RE = re.compile(
    r".*/(\d+)\.(\w+)\.(\d+)\.([\w-]+)\.(\w+)\.xml"
)


@dataclass
class Blog:
    text: str
    date: str = ""

    def truncate(self, max_len: int) -> "Blog":
        """Hard truncate with a marker."""
        if len(self.text) > max_len:
            self.text = (
                self.text[:max_len]
                + "... [post truncated]"
            )
        return self


@dataclass
class Bin:
    gender: str
    low_age: Optional[int] = None
    high_age: Optional[int] = None


def get_blogs(
    filename: str,
    min_length: Optional[int] = None,
) -> List[Blog]:
    """
    Parse an XML blog file and return Blog objects.
    """
    parser = etree.XMLParser(
        recover=True,
        resolve_entities=False,
    )
    tree = etree.parse(filename, parser)

    texts = [x.text.strip() for x in tree.findall("post")]
    dates = [x.text.strip() for x in tree.findall("date")]

    if len(texts) == len(dates):
        blogs = [Blog(t, d) for t, d in zip(texts, dates)]
    else:
        blogs = [Blog(t) for t in texts]

    if min_length is not None:
        blogs = [b for b in blogs if len(b.text) > min_length]

    return blogs


def build_metadata_df(folder: str) -> pd.DataFrame:
    """
    Scan blog XML files and extract user metadata.
    """
    rows = []

    for filename in glob(f"{folder}/*"):
        match = FILENAME_RE.match(filename)
        if not match:
            continue

        blogs = get_blogs(filename)

        rows.append({
            "user_id": match.group(1),
            "gender": match.group(2),
            "age": int(match.group(3)),
            "industry": match.group(4),
            "zodiac": match.group(5),
            "filename": filename,
            "max_length": max(len(b.text) for b in blogs),
        })

    return pd.DataFrame(rows)


def sample_by_bins(
    df: pd.DataFrame,
    num_sample,
    min_blog_len,
    trunc_len,
    seed
) -> pd.DataFrame:
    """
    Sample users evenly across demographic bins and attach one blog per user.
    """
    bins = [
        Bin("male", high_age=19),
        Bin("female", high_age=19),
        Bin("male", low_age=20, high_age=29),
        Bin("female", low_age=20, high_age=29),
        Bin("male", low_age=30),
        Bin("female", low_age=30),
    ]

    samples_per_bin = num_sample // len(bins)
    rng_seed = seed

    df = df[df["max_length"] >= min_blog_len]
    output_rows = []

    for b in bins:
        sub = df[df["gender"] == b.gender]

        if b.low_age is not None:
            sub = sub[sub["age"] >= b.low_age]
        if b.high_age is not None:
            sub = sub[sub["age"] <= b.high_age]

        sampled = sub.sample(
            n=samples_per_bin,
            replace=False,
            random_state=rng_seed,
        )

        for record in sampled.to_dict(orient="records"):
            blogs = get_blogs(
                record["filename"],
                min_length=min_blog_len,
            )

            np.random.seed(rng_seed)
            rng_seed += 1

            blog = np.random.choice(blogs)
            blog.truncate(trunc_len)

            record["blog"] = blog.text
            record["date"] = blog.date

            output_rows.append(record)

    return pd.DataFrame(output_rows)


# =====================
# Main Function
# =====================

def build_blog_dataset(
    data_folder: str="data/blogs",
    output_dir="outputs/data/blogs",
    num_sample=600,
    min_blog_len=300,
    trunc_len=350,
    seed=42
):
    print("Building metadata dataframe...")
    df_meta = build_metadata_df(data_folder)

    print("Sampling blogs...")
    df = sample_by_bins(
        df_meta,
        num_sample=num_sample,
        min_blog_len=min_blog_len,
        trunc_len=trunc_len,
        seed=seed
    )

    texts = [
        [ProfileText(
            text=text,
            metadata={"date": date} if date else None
        )] for (text, date) in zip(df["blog"], df["date"])
    ]

    df.drop(["blog","date"], axis=1, inplace=True)

    dataset = Dataset(
        attribute_df=df,
        texts=texts,
        texts_description="blog posts from blogger.com",
        fields_to_infer=[AGE, GENDER]
    )

    print(f"Saving dataset to directory {output_dir}")
    dataset.save(output_dir)