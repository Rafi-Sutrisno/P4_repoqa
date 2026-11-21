"""Tolerant parsing of an LLM's list of selected IDs."""

import json
import re
from typing import List

_BRACKETS = re.compile(r"\[[^\[\]]*\]")
_INT = re.compile(r"\d+")


def _as_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def parse_ids(text: str, n: int, k: int) -> List[int]:
    """IDs in 1..n from ``text``, in reply order, without duplicates, at most ``k``.

    Tries the first JSON list that contains integers; otherwise takes every integer in the
    text (comma-separated lists, free text). Returns [] when nothing usable is found.
    """
    ids: List[int] = []
    for match in _BRACKETS.finditer(text or ""):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            ids = [i for i in map(_as_int, value) if i is not None]
            if ids:
                break
    if not ids:
        ids = [int(x) for x in _INT.findall(text or "")]
    result: List[int] = []
    for i in ids:
        if 1 <= i <= n and i not in result:
            result.append(i)
    return result[:k]
