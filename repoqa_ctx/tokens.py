"""Token counting with the RepoQA tokenizer (CodeLlama). All budgets use this count."""

from functools import lru_cache

# Same tokenizer as repoqa.search_needle_function.make_code_context.
REPOQA_TOKENIZER = "codellama/CodeLlama-7b-Instruct-hf"


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(REPOQA_TOKENIZER)


def count_tokens(text: str) -> int:
    """Number of tokens as RepoQA counts them (``tokenizer.tokenize``, no special tokens)."""
    if not text:
        return 0
    return len(get_tokenizer().tokenize(text))
