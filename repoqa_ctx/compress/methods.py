"""Compressors for the compress stage (needs torch; run in .venv-compress).

Each compressor takes a task (window, description, language) and a budget B (RepoQA
CodeLlama tokens) and returns (code_context, meta, compress_tokens) with
count_tokens(code_context) <= B. Deviations from the official LongCodeZip RepoQA code
(third_party/LongCodeZip/experiments/repo-qa) are marked "DEVIATION".
"""

import importlib.util
import logging
import os
from typing import Dict, List, Tuple

import torch

from repoqa.search_needle_function import INSTRUCTION
from repoqa_ctx.tokens import count_tokens

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFICIAL_DIR = os.path.join(REPO_ROOT, "third_party", "LongCodeZip", "experiments", "repo-qa")
OFFICIAL_COMMIT = "617a30b"


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# --- official chunkers, copied from LongCodeZip experiments/repo-qa/main.py (MIT) ----------
# main.py:25-32
CHUNK_MARKERS = {
    "python": ["class", "def"],
    "cpp": ["class", "struct", "void", "int", "bool", "double", "float", "char", "auto"],
    "java": ["class", "interface", "void", "int", "boolean", "double", "float", "char"],
    "typescript": ["class", "interface", "function", "const", "let", "var"],
    "rust": ["fn", "struct", "impl", "trait", "enum"],
    "go": ["func", "type", "struct", "interface"],
}


def chunk_by_function(code: str, language: str) -> List[str]:
    """main.py:64-89 (CodeChunker._chunk_by_function)."""
    chunks, current = [], []
    for line in code.split("\n"):
        stripped = line.strip()
        if any(stripped.startswith(m) for m in CHUNK_MARKERS[language]) and current:
            chunks.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_by_sliding_window(code: str, window_size: int, overlap_size: int) -> List[str]:
    """main.py:91-116 (CodeChunker._chunk_by_sliding_window)."""
    lines = code.split("\n")
    if len(lines) <= window_size:
        return ["\n".join(lines)]
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + window_size, len(lines))
        chunks.append("\n".join(lines[start:end]))
        start += window_size - overlap_size
        if len(lines) - start < window_size:
            if len(lines) - start > overlap_size:
                chunks.append("\n".join(lines[start:]))
            break
    return chunks


# --- RAG with UniXcoder --------------------------------------------------------------------
class UnixcoderRAG:
    """main.py:130-162: mean-pooled last hidden state, truncation 512, cosine similarity."""

    def __init__(self, model_name: str, device: str):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def _embed(self, text: str) -> Tuple[torch.Tensor, int]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        return out.last_hidden_state.mean(dim=1).squeeze(), int(inputs["input_ids"].shape[1])

    def compress(self, task: Dict, budget: int, params: Dict, sliding: bool):
        code = task["code_context"]
        if sliding:
            chunks = chunk_by_sliding_window(code, params["window_lines"], params["overlap_lines"])
        else:
            chunks = chunk_by_function(code, task["language"])
        query, used = self._embed(task["description"])  # query = description (main.py:186-189)
        sims = []
        for chunk in chunks:
            emb, n = self._embed(chunk)
            used += n
            sims.append(torch.cosine_similarity(query, emb, dim=0).item())
        ranked = sorted(range(len(chunks)), key=lambda i: sims[i], reverse=True)
        # DEVIATION: official keeps top_k=5 chunks (main.py:151); we add chunks in similarity
        # order while the result fits B. Like the official code, chunks are joined with "\n"
        # in similarity order (main.py:195).
        picked: List[int] = []
        for i in ranked:
            trial = "\n".join(chunks[j] for j in picked + [i])
            if count_tokens(trial) <= budget:
                picked.append(i)
        context = "\n".join(chunks[j] for j in picked)
        return context, {"n_chunks": len(chunks), "n_selected": len(picked)}, used

    def compress_paper(self, task: Dict, params: Dict):
        """Paper version, exactly the official compress_context (main.py:165-207): top-k
        chunks without a token budget, in similarity order, joined with "\\n"."""
        code = task["code_context"]
        if params["chunking"] == "sliding_window":
            chunks = chunk_by_sliding_window(code, params["window_lines"], params["overlap_lines"])
        else:
            chunks = chunk_by_function(code, task["language"])
        query, used = self._embed(task["description"])
        sims = []
        for chunk in chunks:
            emb, n = self._embed(chunk)
            used += n
            sims.append(torch.cosine_similarity(query, emb, dim=0).item())
        # main.py:160-162: sort by similarity (stable for ties), keep the first top_k.
        ranked = sorted(range(len(chunks)), key=lambda i: sims[i], reverse=True)[: params["top_k"]]
        context = "\n".join(chunks[i] for i in ranked)
        return context, {"n_chunks": len(chunks), "n_selected": len(ranked)}, used


# --- LongCodeZip (official CodeCompressor, rank-only) --------------------------------------
def _load_official_code_compressor():
    path = os.path.join(OFFICIAL_DIR, "code_compressor.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing; run scripts/setup_compress.sh (clones LongCodeZip @ {OFFICIAL_COMMIT})"
        )
    spec = importlib.util.spec_from_file_location("lcz_code_compressor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The official module calls logging.basicConfig(level=DEBUG); keep our logs readable.
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("CodeCompressor").setLevel(logging.WARNING)
    return module.CodeCompressor


class LongCodeZipRankOnly:
    def __init__(self, model_name: str, device: str):
        compressor_cls = _load_official_code_compressor()
        # Official default dtype is float16 (code_compressor.py:67); float32 on CPU.
        config = {"torch_dtype": torch.float32} if device == "cpu" else {}
        self.cc = compressor_cls(model_name, device_map=device, model_config=config)

    @staticmethod
    def _render(chunks: List[str], selected, language: str) -> str:
        # Same rendering as the official rank-only path (code_compressor.py:1259-1280).
        marker = "#" if language.lower() in ["python", "typescript", "rust"] else "//"
        return "\n\n".join(c if i in selected else f"{marker} ... " for i, c in enumerate(chunks))

    def compress(self, task: Dict, budget: int, params: Dict):
        code = task["code_context"].replace("<|endoftext|>", "")  # main.py:321-323
        query = task["description"]  # q = description (main.py:331)
        lang = task["language"]
        # target_token=B in the compressor's tokens; context_budget "+100", condition
        # "prefix", original order: official defaults (code_compressor.py:1242-1250).
        # DEVIATION: the official run.sh uses --compression-ratio 0.1-0.4, not a token budget.
        result = self.cc.compress_code_file(
            code=code, query=query, instruction=INSTRUCTION, target_token=budget,
            language=lang, rank_only=True,
        )  # fmt: skip
        chunks = self.cc.split_code_by_functions(code, language=lang)
        selected = list(result["selected_functions"])
        order = [i for i, _ in result["demonstrations_sort"]]
        context = self._render(chunks, set(selected), lang)
        matches_official = context == result["compressed_code"]
        # DEVIATION: the official selection may exceed the budget (+100 slack and one chunk
        # of overshoot, code_compressor.py:1138-1153) and counts the compressor's tokens. We
        # drop the lowest-ranked selected chunks until the context fits B (CodeLlama).
        dropped = 0
        while count_tokens(context) > budget and selected:
            worst = max(selected, key=order.index)
            selected.remove(worst)
            dropped += 1
            context = self._render(chunks, set(selected), lang)
        used = sum(self.cc.get_token_length(c + "\n\n" + query) for c in chunks)
        used += self.cc.get_token_length(query)
        meta = {
            "n_chunks": len(chunks),
            "n_selected": len(selected),
            "dropped_for_budget": dropped,
            "matches_official_render": matches_official,
        }
        return context, meta, used


# --- LLMLingua family ----------------------------------------------------------------------
class LLMLinguaFamily:
    def __init__(self, kind: str, model_name: str, device: str):
        from llmlingua import PromptCompressor

        self.kind = kind
        if kind == "llmlingua2":
            self.pc = PromptCompressor(model_name=model_name, use_llmlingua2=True, device_map=device)
        else:
            self.pc = PromptCompressor(model_name=model_name, device_map=device)

    def _call(self, task: Dict, target: int, params: Dict) -> str:
        code = task["code_context"].replace("<|endoftext|>", "")  # main.py:221-222
        question = task["description"] + INSTRUCTION  # main.py:231
        if self.kind == "llmlingua2":
            # LLMLingua-2 is not in the official RepoQA code; default force_tokens from the
            # llmlingua README example (["\n", "?"]); a variant drops "\n".
            res = self.pc.compress_prompt(
                code, target_token=target, force_tokens=list(params["force_tokens"])
            )
        elif self.kind == "longllmlingua":
            # Line chunks as called in main.py:612-613 (window 20, overlap 10) with
            # rank_method="longllmlingua" (main.py:279).
            lines = code.split("\n")
            step = params["chunk_lines"] - params["overlap_lines"]
            chunks = [
                "\n".join(lines[i : i + params["chunk_lines"]])
                for i in range(0, len(lines), step)
                if lines[i : i + params["chunk_lines"]]
            ]
            res = self.pc.compress_prompt(
                chunks, instruction="", question=question, target_token=target,
                rank_method="longllmlingua", concate_question=False,
            )  # fmt: skip
        else:
            res = self.pc.compress_prompt(
                code, instruction="", question=question, target_token=target,
                concate_question=False,
            )  # fmt: skip
        # DEVIATION: the official call passes instruction=INSTRUCTION and keeps the question
        # in the compressed prompt (main.py:228-233); we pass no instruction and do not
        # append the question, so the output is the compressed code only and goes into the
        # RepoQA template like every other strategy.
        return res["compressed_prompt"]

    def compress(self, task: Dict, budget: int, params: Dict):
        target = budget
        tries = []
        context = ""
        for _ in range(4):
            context = self._call(task, target, params)
            n = count_tokens(context)
            tries.append({"target_token": target, "tokens": n})
            if n <= budget:
                break
            # DEVIATION: target_token counts the compressor's tokens; shrink until it fits B.
            target = max(16, int(target * budget / n * 0.95))
        truncated = False
        while count_tokens(context) > budget and "\n" in context:
            context = context.rsplit("\n", 1)[0]  # last resort: drop trailing lines
            truncated = True
        used = len(self.pc.tokenizer.encode(task["code_context"]))  # approx. one pass
        return context, {"tries": tries, "truncated": truncated}, used


def make_compressor(strategy: str, params: Dict, device: str):
    model = params["compressor_model"]
    if strategy == "rag_paper_unixcoder":
        rag = UnixcoderRAG(model, device)
        return lambda task, budget: rag.compress_paper(task, params)
    if strategy in ("rag_fn_unixcoder", "rag_sw_unixcoder"):
        rag = UnixcoderRAG(model, device)
        sliding = strategy == "rag_sw_unixcoder"
        return lambda task, budget: rag.compress(task, budget, params, sliding)
    if strategy == "longcodezip":
        lcz = LongCodeZipRankOnly(model, device)
        return lambda task, budget: lcz.compress(task, budget, params)
    if strategy in ("llmlingua", "llmlingua2", "longllmlingua"):
        fam = LLMLinguaFamily(strategy, model, device)
        return lambda task, budget: fam.compress(task, budget, params)
    raise ValueError(f"{strategy} is not a compress-stage strategy")
