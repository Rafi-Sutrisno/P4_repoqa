"""Okapi BM25 with an identifier-aware tokenizer (camelCase / snake_case split)."""

import math
import re
from collections import Counter
from typing import List

_WORD = re.compile(r"[A-Za-z0-9]+")
_SUBWORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def tokenize(text: str) -> List[str]:
    out = []
    for word in _WORD.findall(text):
        for part in _SUBWORD.findall(word):
            if len(part) > 1:
                out.append(part.lower())
    return out


def bm25_scores(query: List[str], docs: List[List[str]], k1: float, b: float) -> List[float]:
    n = len(docs)
    if n == 0:
        return []
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df = Counter()
    for d in docs:
        df.update(set(d))
    scores = []
    for d in docs:
        tf = Counter(d)
        norm = k1 * (1 - b + b * len(d) / avgdl)
        s = 0.0
        for term in query:
            if term in tf:
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                s += idf * tf[term] * (k1 + 1) / (tf[term] + norm)
        scores.append(s)
    return scores
