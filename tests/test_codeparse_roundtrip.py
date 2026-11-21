"""split_window must be lossless, and find common function constructs, including in
windows that start in the middle of a file."""

import pytest

from repoqa_ctx.codeparse import join_units, split_window
from repoqa_ctx.tasks import build_tasks


def test_roundtrip_real_windows(mini_dataset, task_cache, monkeypatch):
    from repoqa_ctx import tasks as T

    mini, _ = mini_dataset
    monkeypatch.setattr(T, "TASK_CACHE_DIR", task_cache)
    for task in build_tasks(mini):
        units = split_window(task["code_context"], task["language"])
        assert join_units(units) == task["code_context"]
        assert any(u.kind == "function" for u in units)


CASES = [
    # (language, code, function names that must become units)
    (
        "python",
        '    """end of a docstring that started before the window\n'
        "    more text\n"
        '    """\n'
        "    def method_one(self):\n"
        "        return 1\n"
        "\n"
        "    def method_two(self):\n"
        "        return 2\n",
        {"method_one", "method_two"},
    ),
    (
        "java",
        "        return x;\n"
        "    }\n"
        "\n"
        "    public Foo(int a) {\n"
        "        this.a = a;\n"
        "    }\n"
        "\n"
        "    int bar() { return a; } // trailing comment\n"
        "}\n",
        {"Foo", "bar"},
    ),
    (
        "typescript",
        "// Path: src/util.ts\n"
        "export const add = (a: number, b: number): number => a + b;\n"
        "export async function load(path: string) {\n"
        "  return path;\n"
        "}\n"
        "class Box {\n"
        "  handle = () => { return 1; };\n"
        "  open() { return 2; }\n"
        "}\n",
        {"add", "load", "handle", "open"},
    ),
    (
        "cpp",
        "// Path: src/a.cpp\n"
        "template <typename T> T twice(T x) { return x + x; }\n"
        "int Widget::size() const { return n_; }\n"
        "class C {\n"
        "  friend void swap(C &a, C &b) { }\n"
        "};\n",
        {"twice", "Widget::size", "swap"},
    ),
    (
        "rust",
        "// Path: src/lib.rs\n"
        "impl Point {\n"
        "    pub fn norm(&self) -> f64 { 0.0 }\n"
        "}\n"
        "fn main() {}\n",
        {"norm", "main"},
    ),
    (
        "go",
        "usage text inside a raw string that started before the window\n"
        "`\n"
        "\n"
        "func (s *Server) Start() error {\n"
        "\treturn nil\n"
        "}\n",
        {"Start"},
    ),
]


@pytest.mark.parametrize("language, code, names", CASES, ids=[c[0] for c in CASES])
def test_roundtrip_constructs(language, code, names):
    units = split_window(code, language)
    assert join_units(units) == code
    assert names <= {u.name for u in units if u.kind == "function"}
