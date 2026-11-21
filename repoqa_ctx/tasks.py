"""Build RepoQA SNF tasks exactly like ``repoqa.search_needle_function.evaluate_model``.

The task loop below mirrors evaluate_model (lines 458-522 of the original file) and reuses
``make_code_context`` unchanged. ``tests/test_full_identity.py`` checks that the result is
identical to what the original code produces.
"""

import gzip
import hashlib
import json
import os
import random
import urllib.request
from typing import Dict, List, Optional

from repoqa.data import CACHE_DIR, REPOQA_DATA_OVERRIDE_PATH, REPOQA_DATA_VERSION
from repoqa.search_needle_function import (
    INSTRUCTION,
    TEMPLATE,
    CleanComment,
    make_cache_id,
    make_code_context,
    make_task_id,
)
from repoqa.utility import topological_sort

# Keys of a task dict as produced by the original evaluate_model loop.
ORIGINAL_KEYS = (
    "repo",
    "name",
    "language",
    "path",
    "position_ratio",
    "description",
    "instruction",
    "template",
    "needle_token_start",
    "needle_token_end",
    "code_context_ntokens",
    "code_context",
)

# Separate from the original cache file so the two pipelines never share state.
TASK_CACHE_DIR = CACHE_DIR
SPLIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.json")


def load_dataset(dataset_path: Optional[str] = None) -> Dict:
    """Load the RepoQA dataset.

    Same URL, version, override variable and cache path as ``repoqa.data``, but the file is
    read and written as UTF-8. The original uses the platform default encoding, which fails
    on Windows (cp1252) because the dataset contains non-ASCII code.
    """
    if dataset_path is None and REPOQA_DATA_OVERRIDE_PATH:
        dataset_path = REPOQA_DATA_OVERRIDE_PATH
    if dataset_path is None:
        dataset_path = os.path.join(CACHE_DIR, f"repoqa-{REPOQA_DATA_VERSION}.json")
        if not os.path.exists(dataset_path):
            url = (
                "https://github.com/evalplus/repoqa_release/releases/download/"
                f"{REPOQA_DATA_VERSION}/repoqa-{REPOQA_DATA_VERSION}.json.gz"
            )
            print(f"Downloading dataset from {url}")
            with urllib.request.urlopen(url) as resp:
                text = gzip.decompress(resp.read()).decode("utf-8")
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp_path = dataset_path + ".part"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp_path, dataset_path)
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def make_uid(task: Dict) -> str:
    """Opaque task id; strategies never see the needle name or path."""
    task_id = make_task_id(task["language"], task["repo"], task["name"])
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]


def build_tasks(
    dataset: Dict,
    code_context_size: int = 16 * 1024,
    languages: Optional[List[str]] = None,
    limit: Optional[int] = None,
    cache_dir: Optional[str] = None,
    caching: bool = True,
) -> List[Dict]:
    """Return tasks in the original order; each task gets an extra ``uid`` field."""
    cache_dir = cache_dir or TASK_CACHE_DIR
    cache_file = os.path.join(
        cache_dir, f"repoqa_ctx_tasks_ntoken_{code_context_size}_v1.jsonl"
    )
    cache = {}
    if caching and os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a truncated last line
                cache[item.pop("cache_id")] = item

    tasks = []
    for lang, repos in dataset.items():
        if languages is not None and lang not in languages:
            continue
        for repo in repos:
            if "needles" not in repo:
                continue
            ordered_paths = topological_sort(repo["dependency"])
            file_content_list = [(path, repo["content"][path]) for path in ordered_paths]
            for i, needle in enumerate(repo["needles"]):
                if limit is not None and len(tasks) >= limit:
                    return tasks
                position_ratio = (i + 0.5) / len(repo["needles"])
                cache_id = make_cache_id(
                    lang, repo["repo"], needle["name"], code_context_size, position_ratio
                )
                if cache_id in cache:
                    task = cache[cache_id]
                else:
                    task = {
                        "repo": repo["repo"],
                        "name": needle["name"],
                        "language": lang,
                        "path": needle["path"],
                        "position_ratio": position_ratio,
                        "description": f"\nFunction Description:{needle['description']}\n",
                        "instruction": INSTRUCTION,
                        "template": TEMPLATE,
                    }
                    task.update(
                        make_code_context(
                            needle,
                            file_content_list,
                            position_ratio=position_ratio,
                            code_context_size=code_context_size,
                            language=lang,
                            clean_comments=CleanComment.NoClean,
                        )
                    )
                    if caching:
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(cache_file, "a", encoding="utf-8") as f_out:
                            f_out.write(json.dumps({"cache_id": cache_id, **task}) + "\n")
                task = dict(task)
                task["uid"] = make_uid(task)
                tasks.append(task)
    return tasks


def gold_functions(dataset: Dict) -> Dict[str, str]:
    """uid -> ground-truth needle text, taken line-wise exactly like needle_evaluator.

    Only the runner uses this (needle retention, fake oracle); strategies never do.
    """
    gold = {}
    for lang, repos in dataset.items():
        for repo in repos:
            for needle in repo.get("needles", []):
                lines = repo["content"][needle["path"]].split("\n")
                text = "\n".join(lines[needle["start_line"] : needle["end_line"]])
                uid = make_uid({"language": lang, "repo": repo["repo"], "name": needle["name"]})
                gold[uid] = text
    return gold


def make_split(dataset: Dict, seed: int = 0) -> Dict:
    """Dev split = one repo per language, chosen with a fixed seed; the rest is test."""
    rng = random.Random(seed)
    dev = {}
    for lang in sorted(dataset):
        names = sorted(r["repo"] for r in dataset[lang] if "needles" in r)
        dev[lang] = rng.choice(names)
    return {"seed": seed, "dev_repos": dev}


def load_split() -> Dict:
    with open(SPLIT_PATH, encoding="utf-8") as f:
        return json.load(f)


def split_of(task: Dict, split: Dict) -> str:
    return "dev" if split["dev_repos"].get(task["language"]) == task["repo"] else "test"


if __name__ == "__main__":
    # Writes repoqa_ctx/split.json once; never overwrites an existing split.
    if os.path.exists(SPLIT_PATH):
        raise SystemExit(f"{SPLIT_PATH} already exists; refusing to overwrite")
    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(make_split(load_dataset()), f, indent=2)
        f.write("\n")
    print(f"Wrote {SPLIT_PATH}")
