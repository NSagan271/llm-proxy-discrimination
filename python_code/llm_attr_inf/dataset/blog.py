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
import unicodedata

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

    def truncate(self, target_len: int, max_len: int) -> "Blog":
        self.text = truncate_blog_post(
            self.text,
            target_length=target_len,
            max_length=max_len,
        )
        return self


@dataclass
class Bin:
    gender: str
    low_age: Optional[int] = None
    high_age: Optional[int] = None



CSS_RE = re.compile(r'\{[^}]*\}')
HTML_RE = re.compile(r'<[^>]+>')


def clean_post(text: str) -> str:
    text = HTML_RE.sub(' ', text)
    text = CSS_RE.sub(' ', text)
    text = re.sub(r'urlLink\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def english_char_ratio(text: str) -> float:
    if not text:
        return 0.0

    english = 0
    total = 0

    for ch in text:
        if ch.isalpha():
            total += 1
            name = unicodedata.name(ch, "")
            if "LATIN" in name:
                english += 1

    return english / total if total > 0 else 0.0


def is_mostly_english(text, threshold=0.9):
    return english_char_ratio(text) >= threshold


def has_too_many_replacement_chars(text, max_frac=0.01):
    if not text:
        return True
    return text.count("�") / len(text) > max_frac


def clean_texts(texts):
    cleaned_texts = []
    cleaned_idxs = []

    for i, raw in enumerate(texts):
        raw = raw.strip()

        text = clean_post(raw)

        if has_too_many_replacement_chars(text):
            continue

        if not is_mostly_english(text, threshold=0.9):
            continue
        cleaned_idxs.append(i)
        cleaned_texts.append(text)
    return cleaned_texts, cleaned_idxs

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

    texts, idxs = clean_texts([x.text for x in tree.findall("post")])

    dates = [x.text.strip() for x in tree.findall("date")]
    dates = [dates[i] for i in idxs]

    blogs = [Blog(t, d) for t, d in zip(texts, dates)]

    if min_length is not None:
        blogs = [b for b in blogs if len(b.text) > min_length]

    return blogs



def truncate_blog_post(
    text: str,
    target_length: int,
    max_length: int,
    suffix: str = ""
) -> str:
    """
    Truncate text according to the following rules:

    1) Find the first sentence or clause boundary after target_length
       but before max_length, and cut there.
    2) Otherwise, find the first whitespace after target_length
       but before max_length, and cut there.
    3) Otherwise, cut exactly at max_length.

    If truncation occurs, append `suffix`. If not, return the original text.
    """
    n = len(text)

    # If already short enough, do nothing
    if n <= max_length:
        return text

    # Ensure bounds make sense
    target_length = max(0, min(target_length, n))
    max_length = max(0, min(max_length, n))

    window = text[target_length:max_length]

    # 1) Sentence or clause boundaries
    # Includes sentence end and strong clause punctuation
    sentence_boundary_pattern = re.compile(r"[.!?;:]\s")
    match = sentence_boundary_pattern.search(window)
    if match:
        cut_idx = target_length + match.end() - 1
        return text[:cut_idx].rstrip() + suffix

    # 2) Whitespace boundary
    whitespace_match = re.search(r"\s", window)
    if whitespace_match:
        cut_idx = target_length + whitespace_match.start()
        return text[:cut_idx].rstrip() + suffix

    # 3) Hard cutoff
    return text[:max_length].rstrip() + suffix


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
        if len(blogs) == 0:
            # all of the blogs got filtered out
            continue

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
    target_len,
    max_len,
    seed
) -> pd.DataFrame:
    """
    Sample users evenly across demographic bins and attach one blog per user.
    """
    bins = [
        Bin("male", low_age=18, high_age=29),
        Bin("female", low_age=18, high_age=29),
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

            blog: Blog = np.random.choice(blogs)
            blog.truncate(
                target_len=target_len,
                max_len=max_len,
            )

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
    target_length=350,
    max_length=400,
    seed=42
):
    print("Building metadata dataframe...")
    df_meta = build_metadata_df(data_folder)

    print("Sampling blogs...")
    df = sample_by_bins(
        df_meta,
        num_sample=num_sample,
        min_blog_len=min_blog_len,
        target_len=target_length,
        max_len=max_length,
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
        texts_description="the first part of a blog post",
        fields_to_infer=[AGE, GENDER],
        extra_description="The rest of the post will be cut off, maybe in the middle of a sentence."
    )

    print(f"Saving dataset to directory {output_dir}")
    dataset.save(output_dir)