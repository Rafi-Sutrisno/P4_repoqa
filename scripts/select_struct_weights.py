"""Pick struct_bm25 field weights on the DEV split only (no LLM): needle retention.

    python scripts/select_struct_weights.py [--out reports/struct_weights_dev.md]

Candidates were fixed before looking at any result (name, signature, doc, body):
    [1,1,1,1] reference, [2,1,1,2], [3,1,1,3], [2,1,1,3]
Score = mean retention over B in {1024, 2048, 4096}; ties go to the earlier candidate.
"""

import argparse
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repoqa_ctx import tasks as T  # noqa: E402
from repoqa_ctx.run import needle_retained  # noqa: E402
from repoqa_ctx.strategies import struct_bm25  # noqa: E402
from repoqa_ctx.strategies.base import StrategyInput  # noqa: E402

CANDIDATES = [[1, 1, 1, 1], [2, 1, 1, 2], [3, 1, 1, 3], [2, 1, 1, 3]]
BUDGETS = [1024, 2048, 4096]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join("reports", "struct_weights_dev.md"))
    args = p.parse_args()

    dataset = T.load_dataset()
    split = T.load_split()
    gold = T.gold_functions(dataset)
    dev = [t for t in T.build_tasks(dataset) if T.split_of(t, split) == "dev"]

    rows = []
    for weights in CANDIDATES:
        ret = {}
        for budget in BUDGETS:
            kept = []
            for t in dev:
                inp = StrategyInput(
                    t["uid"], t["language"], t["code_context"], t["description"],
                    budget, {"weights": weights}, 0,
                )  # fmt: skip
                out = struct_bm25.build_context(inp)
                kept.append(needle_retained(gold[t["uid"]], out.code_context))
            ret[budget] = mean(kept)
        rows.append((weights, ret, mean(ret.values())))
        print(weights, {b: round(100 * r, 1) for b, r in ret.items()})

    best = max(rows, key=lambda r: r[2])  # max keeps the first of equal scores
    lines = [
        "# Pemilihan bobot struct_bm25 (split dev saja)",
        "",
        f"Split dev: {len(dev)} tugas (1 repo per bahasa, `repoqa_ctx/split.json`). Tanpa LLM; "
        "ukuran = retensi needle. Kandidat ditetapkan sebelum melihat hasil; urutan bobot: "
        "nama, signature, doc, isi. Skor = rata-rata retensi pada B 1024/2048/4096; "
        "seri dimenangkan kandidat yang lebih awal.",
        "",
        "| bobot | B=1024 | B=2048 | B=4096 | rata-rata |",
        "|---|---|---|---|---|",
    ]
    for weights, ret, avg in rows:
        cells = " | ".join(f"{100 * ret[b]:.1f}%" for b in BUDGETS)
        mark = " **(terpilih)**" if weights == best[0] else ""
        lines.append(f"| {weights}{mark} | {cells} | {100 * avg:.1f}% |")
    lines += ["", f"Terpilih: `{{\"weights\": {best[0]}}}`.", ""]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("selected", best[0], "->", args.out)


if __name__ == "__main__":
    main()
