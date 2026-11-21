"""LLM clients (Ollama, fake) and a logger that writes one JSON line per LLM call."""

import json
import os
import re
import textwrap
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import requests

from repoqa.provider.request import construct_message_list
from repoqa_ctx.tokens import count_tokens


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int  # RepoQA (CodeLlama) count of system + user text
    completion_tokens: int  # RepoQA (CodeLlama) count of the reply
    prompt_eval_count: Optional[int] = None  # as reported by the backend
    eval_count: Optional[int] = None
    latency_s: float = 0.0  # wall clock of the HTTP call
    timings: Dict[str, float] = field(default_factory=dict)


class LLMClient:
    model: str = ""

    def chat(
        self, prompt: str, *, max_tokens: int, system: Optional[str] = None, stage: str = "answer"
    ) -> LLMResult:
        raise NotImplementedError


MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")


def model_num_ctx(model: Optional[str]) -> int:
    """num_ctx for a model from repoqa_ctx/models.json (falls back to the default)."""
    with open(MODELS_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["num_ctx"].get(model, cfg["default_num_ctx"])


def _count_prompt(prompt: str, system: Optional[str]) -> int:
    return count_tokens(prompt) + (count_tokens(system) if system else 0)


class OllamaClient(LLMClient):
    """Native Ollama /api/chat with explicit options (no SIGALRM, works on Windows)."""

    def __init__(
        self,
        model: str,
        url: str = "http://localhost:11434",
        num_ctx: int = 24576,
        seed: int = 0,
        timeout: float = 900.0,
        retries: int = 3,
    ):
        self.model = model
        self.url = url.rstrip("/")
        self.num_ctx = num_ctx
        self.seed = seed
        self.timeout = timeout
        self.retries = retries

    def options(self, max_tokens: int) -> Dict:
        return {
            "temperature": 0.0,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": max_tokens,
            "repeat_penalty": 1.0,
        }

    def _post(self, path: str, payload: Dict) -> Dict:
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(self.url + path, json=payload, timeout=self.timeout)
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"{resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:  # client errors (e.g. unknown model): do not retry
                    raise RuntimeError(f"Ollama {path} -> {resp.status_code}: {resp.text[:500]}")
                return resp.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                if attempt == self.retries:
                    raise
                wait = 5 * (attempt + 1)
                print(f"Ollama request failed ({e}); retrying in {wait}s")
                time.sleep(wait)

    def chat(
        self, prompt: str, *, max_tokens: int, system: Optional[str] = None, stage: str = "answer"
    ) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": construct_message_list(prompt, system),
            "stream": False,
            "options": self.options(max_tokens),
        }
        start = time.perf_counter()
        data = self._post("/api/chat", payload)
        latency = time.perf_counter() - start
        text = data["message"]["content"]
        timings = {
            key: data.get(key, 0) / 1e9
            for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")
        }
        return LLMResult(
            text=text,
            prompt_tokens=_count_prompt(prompt, system),
            completion_tokens=count_tokens(text),
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
            latency_s=latency,
            timings=timings,
        )

    def version(self) -> str:
        resp = requests.get(self.url + "/api/version", timeout=30)
        resp.raise_for_status()
        return resp.json()["version"]

    def ps(self) -> Optional[Dict]:
        """Loaded-model status from /api/ps (size, size_vram, ...) for this model."""
        resp = requests.get(self.url + "/api/ps", timeout=30)
        resp.raise_for_status()
        for entry in resp.json().get("models", []):
            if self.model in (entry.get("name"), entry.get("model")):
                info = {k: entry.get(k) for k in ("size", "size_vram", "context_length")}
                if info["size"]:
                    info["vram_share"] = round((info["size_vram"] or 0) / info["size"], 3)
                return info
        return None


def _nows(text: str) -> str:
    return "".join(text.split())


class FakeClient(LLMClient):
    """Deterministic client for pipeline tests and dry-runs.

    The caller sets ``gold`` (ground-truth needle text) before each task.
    mode="oracle":       answer = the gold function in a code block; select = IDs of index
                         lines whose signature occurs in the gold text (longest first)
    mode="select_first": select = [1, 2, ..., 20]; answer = ""
    mode="empty":        always ""
    Other stages: confidence (S4) = {"id", "confidence"}; hyde (S5) = guessed code (the gold
    function for oracle); verify (S6) = "A"/"B" (the candidate holding the gold for oracle);
    index_build (X1-sum) = a fixed one-sentence summary.
    Every prompt is kept in ``prompts`` so tests can inspect what would be sent.
    """

    MODES = ("oracle", "select_first", "empty")

    def __init__(self, mode: str = "oracle"):
        if mode not in self.MODES:
            raise ValueError(f"unknown fake mode: {mode}")
        self.mode = mode
        self.model = f"fake-{mode}"
        self.gold: Optional[str] = None
        self.prompts: List[str] = []

    def _oracle_ids(self, prompt: str) -> List[int]:
        gold = _nows(self.gold or "")
        hits = []
        for match in re.finditer(r"^\[(\d+)\] (.*)$", prompt, re.M):
            signature = _nows(match.group(2).split(": ", 1)[-1])
            if signature and signature in gold:
                hits.append((-len(signature), int(match.group(1))))
        return [number for _, number in sorted(hits)]

    def _oracle_choice(self, prompt: str) -> str:
        gold = _nows(self.gold or "")
        parts = re.split(r"^Candidate ([AB]):$", prompt, flags=re.M)
        for label, body in zip(parts[1::2], parts[2::2]):
            if gold and gold in _nows(body):
                return label
        return "A"

    def _reply(self, prompt: str, stage: str) -> str:
        if self.mode == "empty":
            return ""
        if stage == "select":
            if self.mode == "select_first":
                return json.dumps(list(range(1, 21)))
            return json.dumps(self._oracle_ids(prompt))
        if stage == "confidence":  # S4 adaptive_k (llm)
            ids = self._oracle_ids(prompt) if self.mode == "oracle" else [1]
            level = "high" if self.mode == "oracle" and ids else "medium"
            return json.dumps({"id": ids[0] if ids else 1, "confidence": level})
        if stage == "hyde":  # S5: a guessed implementation
            if self.mode == "oracle" and self.gold is not None:
                return "```\n" + self.gold + "\n```"
            return "```\ndef guess():\n    pass\n```"
        if stage == "verify":  # S6: "A" or "B"
            return self._oracle_choice(prompt) if self.mode == "oracle" else "A"
        if stage == "index_build":  # X1-sum: one-sentence summary
            return "Summary of the function."
        if self.mode == "oracle" and self.gold is not None:
            return "```\n" + textwrap.dedent(self.gold) + "\n```"
        return ""

    def chat(
        self, prompt: str, *, max_tokens: int, system: Optional[str] = None, stage: str = "answer"
    ) -> LLMResult:
        self.prompts.append(prompt)
        text = self._reply(prompt, stage)
        return LLMResult(
            text=text,
            prompt_tokens=_count_prompt(prompt, system),
            completion_tokens=count_tokens(text),
        )


class CallLogger:
    """Wraps a client; appends exactly one JSON line per LLM call to ``path``.

    ``context`` (e.g. uid, strategy) is copied into each line. ``task_calls`` collects the
    lines of the current task so the runner can store them with the task record.
    """

    def __init__(self, client: LLMClient, path: str):
        self.client = client
        self.path = path
        self.context: Dict = {}
        self.task_calls: List[Dict] = []

    @property
    def model(self) -> str:
        return self.client.model

    def chat(
        self, prompt: str, *, stage: str, max_tokens: int, system: Optional[str] = None
    ) -> LLMResult:
        line = {
            "time": datetime.now().isoformat(timespec="seconds"),
            **self.context,
            "stage": stage,
            "model": self.client.model,
        }
        try:
            result = self.client.chat(prompt, max_tokens=max_tokens, system=system, stage=stage)
        except Exception as e:
            line.update({"ok": False, "error": repr(e)})
            self._write(line)
            raise
        info = asdict(result)
        info.pop("text")
        line.update({**info, "ok": True})
        self._write(line)
        self.task_calls.append(line)
        return result

    def _write(self, line: Dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
