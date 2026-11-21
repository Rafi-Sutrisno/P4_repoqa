"""Strategy interface.

A strategy module defines:
    NEEDS_BUDGET: bool
    DEFAULT_PARAMS: dict
    def build_context(inp: StrategyInput, llm) -> StrategyOutput

Strategies only see an opaque id, the language, the code window and the description.
They never see the needle name, path or position.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class StrategyInput:
    uid: str
    language: str
    code_context: str  # window from make_code_context
    description: str  # "\nFunction Description:...\n"
    budget: Optional[int]  # max tokens of the returned code_context (RepoQA tokenizer)
    params: Dict
    seed: int

    def rng(self) -> random.Random:
        # String seeds are hashed with SHA-512, so this is stable across processes.
        return random.Random(f"{self.seed}:{self.uid}")


@dataclass
class StrategyOutput:
    code_context: str  # goes into the original RepoQA template
    meta: Dict = field(default_factory=dict)


def merge_params(defaults: Dict, given: Dict) -> Dict:
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError(f"unknown strategy args: {sorted(unknown)}; allowed: {sorted(defaults)}")
    return {**defaults, **given}
