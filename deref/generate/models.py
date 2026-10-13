"""Models used for generation: Hugging Face ids, pinned revisions and the token id of `</think>`.

Weights and tokenizers are loaded by repository id and revision (resolved through the Hugging Face cache, `HF_HOME`).
"""

QWEN3_8_27B = {
    "repo_id": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "think_end": 248069,  # `</think>`; not a special token, so it stays in the decoded output
}
QWEN3_8B = {
    "repo_id": "Qwen/Qwen3-8B",
    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "think_end": 151668,  # `</think>` in the Qwen3-8B tokenizer
}
MODELS = {"qwen3_8_27b": QWEN3_8_27B, "qwen3_8b": QWEN3_8B}


def load_tokenizer(name):
    """Tokenizer of a model in MODELS (by repository id and pinned revision)."""
    from transformers import AutoTokenizer
    m = MODELS[name]
    return AutoTokenizer.from_pretrained(m["repo_id"], revision=m["revision"])
