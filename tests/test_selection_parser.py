"""Tolerant parsing of selected IDs, and the S1 fallback when nothing usable comes back."""

import pytest

from repoqa_ctx.llm import LLMResult
from repoqa_ctx.selection import parse_ids
from repoqa_ctx.strategies import index_select, struct_bm25
from repoqa_ctx.strategies.base import StrategyInput
from repoqa_ctx.codeparse import split_window


@pytest.mark.parametrize(
    "text, expected",
    [
        ("[3, 17]", [3, 17]),  # JSON
        ('["3", "5"]', [3, 5]),  # JSON with strings
        ("```json\n[2, 4]\n```", [2, 4]),  # JSON in a code block
        ('{"ids": [5, 1]}', [5, 1]),  # JSON object around a list
        ("3, 17, 2", [3, 17, 2]),  # comma-separated
        ("The best match is function 12, then 4.", [12, 4]),  # free text
        ("[0, 3, 99]", [3]),  # out of range
        ("[3, 3, 5]", [3, 5]),  # duplicates
        ("[true, 2]", [2]),  # booleans are not IDs
        ("[1, 2, 3, 4, 5, 6]", [1, 2, 3, 4, 5]),  # at most k
        ("", []),
        ("no idea", []),
    ],
)
def test_parse_ids(text, expected):
    assert parse_ids(text, n=20, k=5) == expected


class _Reply:
    model = "stub"

    def __init__(self, text):
        self.text = text

    def chat(self, prompt, *, stage, max_tokens, system=None):
        assert stage == "select"
        return LLMResult(text=self.text, prompt_tokens=0, completion_tokens=0)


CODE = (
    "# Path: pkg/strings.py\n"
    "def merge_adjacent_strings(parts):\n"
    "    return ''.join(parts)\n"
    "\n"
    "def parse_header(line):\n"
    "    return line.split(':')\n"
)
DESCRIPTION = "\nFunction Description:Merge adjacent strings into one string.\n"


def _input():
    return StrategyInput("u1", "python", CODE, DESCRIPTION, 1024, {}, 0)


def test_selected_ids_are_used():
    out = index_select.build_context(_input(), _Reply("[2]"))
    assert out.meta["picked_ids"] == [2] and not out.meta["fallback"]
    assert "parse_header" in out.code_context and "merge_adjacent" not in out.code_context


def test_unusable_reply_falls_back_to_s1():
    out = index_select.build_context(_input(), _Reply("I cannot tell."))
    assert out.meta["fallback"]
    units = split_window(CODE, "python")
    top = units[struct_bm25.rank_units(DESCRIPTION, units, {})[0]]
    assert top.name == "merge_adjacent_strings" and top.text in out.code_context


from repoqa_ctx.selection import parse_choice, parse_confidence  # noqa: E402


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"id": 7, "confidence": "high"}', (7, "high")),
        ('```json\n{"id": "3", "confidence": "Low"}\n```', (3, "low")),
        ("ID 4, confidence: medium", (4, "medium")),
        ('{"id": 99, "confidence": "high"}', (None, "high")),  # out of range
        ("I think it is 5", (5, None)),
        ("no idea", (None, None)),
    ],
)
def test_parse_confidence(text, expected):
    assert parse_confidence(text, n=20) == expected


@pytest.mark.parametrize(
    "text, expected",
    [("B", "B"), ("Answer: A", "A"), ("Candidate B is the one.", "B"), ("neither", None), ("", None)],
)
def test_parse_choice(text, expected):
    assert parse_choice(text) == expected
