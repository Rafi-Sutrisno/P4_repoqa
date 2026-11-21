"""Split a RepoQA code window into file headers, function units and other code.

``"".join(u.text for u in split_window(ctx, lang)) == ctx`` always holds.

Function units come from a general tree-sitter walk over common syntax (functions, methods,
constructors, arrow functions, qualified C++ definitions, ...), not from RepoQA's
FUNCTION_QUERY, which defines needle eligibility and would leak benchmark construction.
Units cover whole source lines. When a window starts in the middle of a file and the parse
fails, the parser retries from later line starts and with synthetic enclosing context, and
keeps the parse that finds the most function code.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, NamedTuple, Optional, Tuple

from tree_sitter_languages import get_parser

from repoqa.search_needle_function import COMMENT_PREFIX

FUNCTION_NODE_TYPES = {
    "python": {"function_definition"},
    "java": {"method_declaration", "constructor_declaration", "compact_constructor_declaration"},
    "typescript": {
        "function_declaration",
        "generator_function_declaration",
        "method_definition",
    },
    "rust": {"function_item"},
    "cpp": {"function_definition"},
    "go": {"function_declaration", "method_declaration"},
}
# Statements that wrap a function and belong to it.
WRAPPER_NODE_TYPES = {
    "typescript": {"export_statement"},
    "cpp": {"template_declaration", "friend_declaration"},
}
# TS/JS values that make a variable or class field a function.
TS_FUNCTION_VALUES = {"arrow_function", "function", "function_expression", "generator_function"}
COMMENT_NODE_TYPES = {
    "python": {"comment"},
    "java": {"line_comment", "block_comment"},
    "typescript": {"comment"},
    "rust": {"line_comment", "block_comment"},
    "cpp": {"comment"},
    "go": {"comment"},
}
# Keywords right before a function that belong to it. make_code_context inserts a newline
# at the needle start, so "export async function f" becomes "export \nasync function f"
# and tree-sitter no longer sees an export_statement.
LEADING_KEYWORDS = {"typescript": re.compile(rb"export\s+(?:default\s+)?$")}
# Enclosing context for a window that starts inside a class body.
CLASS_PREFIX = {"java": b"class _ {\n", "typescript": b"class _ {\n"}
MAX_RETRY_STARTS = 64
RETRY_SPAN = 0.3  # retry parse starts within the first 30% of the segment


@dataclass(frozen=True)
class Unit:
    kind: str  # "header" | "function" | "code"
    text: str
    segment: int  # index of the file segment (one header and the code under it)
    path: str = ""  # from the segment's Path header; "" if the window starts mid-file
    name: str = ""  # function units only
    signature: str = ""  # function units: text before the body, whitespace collapsed
    doc: str = ""  # function units: docstring, leading comments, comments inside


class _Fn(NamedTuple):
    start: int
    end: int
    name: str
    signature: str
    doc: str


@lru_cache(maxsize=None)
def header_regex(language: str):
    # Header line written by make_code_context: "<comment prefix> Path: <path>\n"
    return re.compile(r"^" + re.escape(COMMENT_PREFIX[language]) + r" Path: [^\n]*\n?", re.M)


def _text(buf: bytes, node) -> str:
    return buf[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _name_node(node, language):
    if language == "cpp":
        decl = node.child_by_field_name("declarator")
        while decl is not None and decl.type != "function_declarator":
            decl = decl.child_by_field_name("declarator")
        return decl.child_by_field_name("declarator") if decl is not None else None
    return node.child_by_field_name("name")


def _inner_function(node, language):
    """(function node, name node) if ``node`` is a function unit, else None."""
    if node.type in FUNCTION_NODE_TYPES[language]:
        return node, _name_node(node, language)
    if language == "typescript":
        if node.type in ("lexical_declaration", "variable_declaration"):
            for child in node.named_children:
                if child.type == "variable_declarator":
                    value = child.child_by_field_name("value")
                    if value is not None and value.type in TS_FUNCTION_VALUES:
                        return value, child.child_by_field_name("name")
        if node.type in ("public_field_definition", "field_definition"):
            value = node.child_by_field_name("value")
            if value is not None and value.type in TS_FUNCTION_VALUES:
                return value, node.child_by_field_name("name") or node.child_by_field_name(
                    "property"
                )
    if node.type in WRAPPER_NODE_TYPES.get(language, ()):
        for child in node.named_children:
            found = _inner_function(child, language)
            if found:
                return found
    return None


def _doc(buf: bytes, unit_node, inner, language) -> str:
    comment_types = COMMENT_NODE_TYPES[language]
    parts = []
    if language == "python":  # docstring = first statement of the body, if a string
        body = inner.child_by_field_name("body")
        if body is not None and body.named_children:
            first = body.named_children[0]
            if first.type == "expression_statement" and first.named_children:
                if first.named_children[0].type == "string":
                    parts.append(_text(buf, first))
    anchor = unit_node
    if anchor.parent is not None and anchor.parent.type == "decorated_definition":
        anchor = anchor.parent
    leading = []
    prev = anchor.prev_sibling
    while prev is not None and prev.type in comment_types:
        leading.append(_text(buf, prev))
        prev = prev.prev_sibling
    parts.extend(reversed(leading))
    inside = []
    stack = [inner]
    while stack:
        node = stack.pop()
        if node.type in comment_types:
            inside.append(node)
            continue
        stack.extend(node.children)
    parts.extend(_text(buf, n) for n in sorted(inside, key=lambda n: n.start_byte))
    return "\n".join(parts)


def _walk(buf: bytes, language: str):
    """Function units of ``buf`` (byte offsets) and whether the parse has errors."""
    tree = get_parser(language).parse(buf)
    found = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        match = _inner_function(node, language)
        if match is None:
            stack.extend(node.children)
            continue  # nested functions stay inside their enclosing unit
        inner, name = match
        body = inner.child_by_field_name("body")
        sig_end = body.start_byte if body is not None else node.end_byte
        signature = " ".join(buf[node.start_byte : sig_end].decode("utf-8", "replace").split())
        found.append(
            _Fn(
                node.start_byte,
                node.end_byte,
                _text(buf, name) if name is not None else "",
                signature,
                _doc(buf, node, inner, language),
            )
        )
    found.sort()
    result, last_end = [], 0
    for fn in found:
        if fn.start >= last_end and fn.end > fn.start:
            result.append(fn)
            last_end = fn.end
    return result, tree.root_node.has_error


def _python_prefix(text: bytes) -> bytes:
    """Open one block per indentation level so indented code at the start parses."""
    indents = []
    for line in text.split(b"\n"):
        if not line.strip():
            continue
        indent = line[: len(line) - len(line.lstrip(b" \t"))]
        if not indent:
            break  # reached top-level code
        indents.append(indent)
    if not indents:
        return b""
    levels = sorted({i for i in indents if len(i) <= len(indents[0])}, key=len)
    prefix, prev = b"", b""
    for level in levels:
        prefix += prev + b"if 1:\n"
        prev = level
    return prefix


def _context_prefixes(text: bytes, language: str) -> List[bytes]:
    if language == "python":
        prefix = _python_prefix(text)
        return [b"", prefix] if prefix else [b""]
    if language in CLASS_PREFIX:
        return [b"", CLASS_PREFIX[language]]
    return [b""]


def _parse_from(data: bytes, language: str, offset: int, prefix: bytes):
    fns, has_error = _walk(prefix + data[offset:], language)
    shift = offset - len(prefix)
    return [
        fn._replace(start=fn.start + shift, end=fn.end + shift)
        for fn in fns
        if fn.start >= len(prefix)
    ], has_error


def _functions(data: bytes, language: str, mid_file: bool) -> List[_Fn]:
    best, has_error = _parse_from(data, language, 0, b"")
    if not (mid_file and has_error):
        return best
    limit = int(len(data) * RETRY_SPAN)
    starts = [0] + [i + 1 for i in range(limit) if data[i : i + 1] == b"\n"]
    if len(starts) > MAX_RETRY_STARTS:
        step = len(starts) / MAX_RETRY_STARTS
        starts = [starts[int(k * step)] for k in range(MAX_RETRY_STARTS)]
    best_cover = sum(fn.end - fn.start for fn in best)
    for offset in starts:
        for prefix in _context_prefixes(data[offset:], language):
            if offset == 0 and not prefix:
                continue
            fns, _ = _parse_from(data, language, offset, prefix)
            cover = sum(fn.end - fn.start for fn in fns)
            if cover > best_cover:
                best, best_cover = fns, cover
    return best


def _whole_lines(data: bytes, fns: List[_Fn], language: str) -> List[_Fn]:
    """Extend units to whole lines without overlapping their neighbours."""
    keyword = LEADING_KEYWORDS.get(language)
    result = []
    prev_end = 0
    for i, fn in enumerate(fns):
        start, end = fn.start, fn.end
        if keyword is not None:
            match = keyword.search(data, prev_end, start)
            if match and match.end() == start:
                start = match.start()
        line_start = data.rfind(b"\n", 0, start) + 1
        if line_start >= prev_end:
            start = line_start
        line_end = data.find(b"\n", end)
        line_end = len(data) if line_end == -1 else line_end
        next_start = fns[i + 1].start if i + 1 < len(fns) else len(data)
        if line_end <= next_start:
            end = line_end
        result.append(fn._replace(start=start, end=end))
        prev_end = end
    return result


def _split_code(text: str, language: str, segment: int, path: str, mid_file: bool) -> List[Unit]:
    data = text.encode("utf-8")
    pieces = []
    pos = 0
    try:
        for fn in _whole_lines(data, _functions(data, language, mid_file), language):
            if fn.start > pos:
                pieces.append(Unit("code", data[pos : fn.start].decode("utf-8"), segment, path))
            pieces.append(
                Unit(
                    "function",
                    data[fn.start : fn.end].decode("utf-8"),
                    segment,
                    path,
                    fn.name,
                    fn.signature,
                    fn.doc,
                )
            )
            pos = fn.end
        if pos < len(data):
            pieces.append(Unit("code", data[pos:].decode("utf-8"), segment, path))
    except UnicodeDecodeError:
        pieces = [Unit("code", text, segment, path)]
    return [u for u in pieces if u.text]


@lru_cache(maxsize=16)
def _split_cached(code_context: str, language: str) -> Tuple[Unit, ...]:
    units: List[Unit] = []
    matches = list(header_regex(language).finditer(code_context))
    segment = 0
    first = matches[0].start() if matches else len(code_context)
    if first > 0:  # the window can start in the middle of a file, without a header
        units += _split_code(code_context[:first], language, segment, "", mid_file=True)
        segment += 1
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(code_context)
        header = match.group(0)
        path = header.split(" Path: ", 1)[1].strip()
        units.append(Unit("header", header, segment, path))
        if match.end() < end:
            body = code_context[match.end() : end]
            units += _split_code(body, language, segment, path, mid_file=False)
        segment += 1
    return tuple(units)


def split_window(code_context: str, language: str) -> List[Unit]:
    return list(_split_cached(code_context, language))


def join_units(units: List[Unit]) -> str:
    return "".join(u.text for u in units)
