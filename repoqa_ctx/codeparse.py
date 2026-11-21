"""Split a RepoQA code window into file headers, function units and other code.

``"".join(u.text for u in split_window(ctx, lang)) == ctx`` always holds.

Function units come from a general tree-sitter walk, not from RepoQA's FUNCTION_QUERY
(which defines needle eligibility and would leak benchmark construction).
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

from tree_sitter_languages import get_parser

from repoqa.search_needle_function import COMMENT_PREFIX

FUNCTION_NODE_TYPES = {
    "python": {"function_definition"},
    "java": {"method_declaration", "constructor_declaration"},
    "typescript": {
        "function_declaration",
        "generator_function_declaration",
        "method_definition",
    },
    "rust": {"function_item"},
    "cpp": {"function_definition"},
    "go": {"function_declaration", "method_declaration"},
}


@dataclass(frozen=True)
class Unit:
    kind: str  # "header" | "function" | "code"
    text: str
    segment: int  # index of the file segment (one header and the code under it)


@lru_cache(maxsize=None)
def header_regex(language: str):
    # Header line written by make_code_context: "<comment prefix> Path: <path>\n"
    return re.compile(r"^" + re.escape(COMMENT_PREFIX[language]) + r" Path: [^\n]*\n?", re.M)


def _function_spans(code: bytes, language: str) -> List[Tuple[int, int]]:
    tree = get_parser(language).parse(code)
    types = FUNCTION_NODE_TYPES[language]
    spans = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in types:
            spans.append((node.start_byte, node.end_byte))
            continue  # nested functions stay inside their enclosing unit
        stack.extend(node.children)
    spans.sort()
    result, last_end = [], 0
    for start, end in spans:
        if start >= last_end and end > start:
            result.append((start, end))
            last_end = end
    return result


def _split_code(text: str, language: str, segment: int) -> List[Unit]:
    data = text.encode("utf-8")
    pieces = []
    pos = 0
    try:
        for start, end in _function_spans(data, language):
            # Attach the indentation before a function that starts its own line.
            s = start
            while s > pos and data[s - 1 : s] in (b" ", b"\t"):
                s -= 1
            if s == 0 or data[s - 1 : s] == b"\n":
                start = s
            if start > pos:
                pieces.append(("code", data[pos:start].decode("utf-8")))
            pieces.append(("function", data[start:end].decode("utf-8")))
            pos = end
        if pos < len(data):
            pieces.append(("code", data[pos:].decode("utf-8")))
    except UnicodeDecodeError:
        pieces = [("code", text)]
    return [Unit(kind, piece, segment) for kind, piece in pieces if piece]


def split_window(code_context: str, language: str) -> List[Unit]:
    units: List[Unit] = []
    matches = list(header_regex(language).finditer(code_context))
    segment = 0
    first = matches[0].start() if matches else len(code_context)
    if first > 0:  # the window can start in the middle of a file, without a header
        units += _split_code(code_context[:first], language, segment)
        segment += 1
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(code_context)
        units.append(Unit("header", match.group(0), segment))
        if match.end() < end:
            units += _split_code(code_context[match.end() : end], language, segment)
        segment += 1
    return units


def join_units(units: List[Unit]) -> str:
    return "".join(u.text for u in units)
