"""Check the lab Ollama setup before running experiments.

    python scripts/lab_check.py [--url http://localhost:11434] [--num-ctx 24576]

For each model it reports:
  - Ollama version and whether the model is installed
  - whether num_ctx takes effect: the same long RepoQA prompt is sent with num_ctx=2048 and
    with --num-ctx; if num_ctx works, only the large setting evaluates the whole prompt
  - our token count (CodeLlama) vs Ollama's prompt_eval_count, plus a repeat of the same
    prompt to see whether prompt caching lowers prompt_eval_count
  - trained context length and how much of the loaded model sits in VRAM
A JSON copy goes to results/lab_check/.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repoqa_ctx import tasks as T  # noqa: E402
from repoqa_ctx.llm import OllamaClient  # noqa: E402
from repoqa_ctx.run import build_prompt  # noqa: E402

MODELS = ["qwen2.5-coder:7b-instruct", "deepseek-coder:6.7b-instruct"]
SMALL_CTX = 2048


def _get(url, path):
    resp = requests.get(url + path, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _post(url, path, payload):
    resp = requests.post(url + path, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def check_model(url, model, num_ctx, long_prompt, max_new_tokens):
    show = _post(url, "/api/show", {"model": model, "name": model})
    info = show.get("model_info", {})
    result = {
        "model": model,
        "quantization": show.get("details", {}).get("quantization_level"),
        "trained_context": next(
            (v for k, v in info.items() if k.endswith(".context_length")), None
        ),
        "calls": [],
    }
    plan = [
        ("short", num_ctx, "Reply with OK."),
        ("long", num_ctx, long_prompt),
        ("long_repeat", num_ctx, long_prompt),
        ("long_small_ctx", SMALL_CTX, long_prompt),
    ]
    for label, ctx, prompt in plan:
        reply = OllamaClient(model, url=url, num_ctx=ctx).chat(prompt, max_tokens=8)
        result["calls"].append(
            {
                "label": label,
                "num_ctx": ctx,
                "our_tokens": reply.prompt_tokens,
                "prompt_eval_count": reply.prompt_eval_count,
                "latency_s": round(reply.latency_s, 2),
                "prompt_eval_s": round(reply.timings.get("prompt_eval_duration", 0), 2),
            }
        )
        if label == "long":
            loaded = [
                m
                for m in _get(url, "/api/ps").get("models", [])
                if model in (m.get("name"), m.get("model"))
            ]
            if loaded:
                result["size_bytes"] = loaded[0].get("size")
                result["size_vram_bytes"] = loaded[0].get("size_vram")
                result["ps_context_length"] = loaded[0].get("context_length")
    by = {c["label"]: c for c in result["calls"]}
    small, big = by["long_small_ctx"]["prompt_eval_count"], by["long"]["prompt_eval_count"]
    result["num_ctx_effective"] = (
        small is not None and big is not None and small <= SMALL_CTX < big
    )
    result["tokens_ratio_ollama_vs_ours"] = round(big / by["long"]["our_tokens"], 3) if big else None
    trained = result["trained_context"]
    result["fits_trained_context"] = (
        None if not (trained and big) else big + max_new_tokens <= trained
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--num-ctx", type=int, default=24576)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    args = p.parse_args()
    url = args.url.rstrip("/")

    report = {"time": datetime.now().isoformat(timespec="seconds"), "url": url}
    report["ollama_version"] = _get(url, "/api/version")["version"]
    installed = [m["name"] for m in _get(url, "/api/tags").get("models", [])]
    report["installed_models"] = installed
    print(f"Ollama {report['ollama_version']} at {url}")

    # A real full-context RepoQA prompt (first task, ~16.6K CodeLlama tokens).
    task = T.build_tasks(T.load_dataset(), limit=1)[0]
    long_prompt = build_prompt(task, task["code_context"])

    report["models"] = []
    for model in args.models:
        if model not in installed:
            print(f"[missing] {model}  ->  run: ollama pull {model}")
            report["models"].append({"model": model, "installed": False})
            continue
        res = check_model(url, model, args.num_ctx, long_prompt, args.max_new_tokens)
        res["installed"] = True
        report["models"].append(res)
        by = {c["label"]: c for c in res["calls"]}
        vram = res.get("size_vram_bytes")
        size = res.get("size_bytes")
        print(f"\n== {model} (quant {res['quantization']}, trained ctx {res['trained_context']})")
        for c in res["calls"]:
            print(
                f"  {c['label']:<15} num_ctx={c['num_ctx']:<6} ours={c['our_tokens']:<6} "
                f"prompt_eval_count={c['prompt_eval_count']} latency={c['latency_s']}s"
            )
        print(f"  num_ctx takes effect: {res['num_ctx_effective']}")
        print(f"  prompt_eval_count / our count (long prompt): {res['tokens_ratio_ollama_vs_ours']}")
        print(
            f"  repeat of same prompt: prompt_eval_count {by['long']['prompt_eval_count']} -> "
            f"{by['long_repeat']['prompt_eval_count']} (lower means prompt caching)"
        )
        print(f"  long prompt + {args.max_new_tokens} new tokens fits trained context: {res['fits_trained_context']}")
        if vram is not None and size:
            print(f"  loaded size {size / 2**30:.1f} GiB, in VRAM {vram / 2**30:.1f} GiB ({100 * vram / size:.0f}%)")

    out_dir = os.path.join("results", "lab_check")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{datetime.now():%Y%m%d-%H%M%S}.json")
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
