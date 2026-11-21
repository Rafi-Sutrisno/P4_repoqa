"""Tolerant parsing of an LLM's list of selected IDs."""

import json
import re
from typing import List, Optional, Tuple

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


_LEVELS = {"high": "high", "medium": "medium", "med": "medium", "low": "low"}
_LEVEL = re.compile(r"\b(high|medium|med|low)\b", re.I)
_CHOICE = re.compile(r"\b([AB])\b")


def parse_confidence(text: str, n: int) -> Tuple[Optional[int], Optional[str]]:
    """(best ID in 1..n or None, "high"|"medium"|"low" or None) from a reply such as
    {"id": 7, "confidence": "high"} or "7, medium" or free text."""
    text = text or ""
    best = None
    try:
        match = re.search(r"\{[^{}]*\}", text)
        value = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        best = _as_int(value.get("id"))
        level = value.get("confidence")
        if isinstance(level, str) and level.lower() in _LEVELS:
            best = best if best is not None and 1 <= best <= n else None
            return best, _LEVELS[level.lower()]
    ids = parse_ids(text, n, 1)
    best = best if best is not None and 1 <= best <= n else (ids[0] if ids else None)
    level = _LEVEL.search(text)
    return best, (_LEVELS[level.group(1).lower()] if level else None)


def parse_choice(text: str) -> Optional[str]:
    """First standalone "A" or "B" in the reply, else None."""
    match = _CHOICE.search(text or "")
    return match.group(1) if match else None
