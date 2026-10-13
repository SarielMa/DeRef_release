"""Judge registry and the Yes/No judge prompt shared by every local (Hugging Face transformers) judge.

Every local judge sees the same chat prompt:
  system: SYSTEM
  user:   f"{context}\n\n{question} Answer Yes or No."
rendered with the model's own chat template (add_generation_prompt=True, enable_thinking=False).
The score is P(Yes) / (P(Yes) + P(No)) over the first answer token (softmax over the two logits at the last position).

Models are given as Hugging Face id + pinned revision; `--model-path` in the entry scripts overrides the location
(a local snapshot directory, loaded with revision=None).
"""
import numpy as np

SYSTEM = "You are a careful evaluator of tool calls made by an AI assistant."

# Qwen3-8B, the encoder of CLM and the tokenizer of the CLM / Jev length checks.
QWEN3_8B = ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")

# name: (HF id, pinned revision, loader, batch size used for truncated (Plain-style) inputs,
#        padded-token budget used for long (full-history) inputs)
MODELS = {
    "qwen3_4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", "causal_lm", 32, 65536),
    "qwen3_8b": (QWEN3_8B[0], QWEN3_8B[1], "causal_lm", 32, 65536),
    "qwen3_8_27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "image_text_to_text", 32, 32768),
    "llama3_1_8b": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659", "causal_lm", 32, 65536),
    "mistral_7b": ("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71", "causal_lm", 32, 65536),
}

# Single-token ids of "Yes" / "No" asserted for the two non-Qwen judges (Qwen judges: asserted single-token only).
YES_NO_IDS = {"llama3_1_8b": (9642, 2822), "mistral_7b": (6360, 2538)}


def source(name, model_path=None):
    """(path or HF id, revision) to load judge `name` from."""
    repo_id, rev = MODELS[name][:2]
    return (model_path, None) if model_path else (repo_id, rev)


def load_tokenizer(name, model_path=None):
    """Tokenizer of judge `name`, left padding. Tokenizers without a pad token (Llama, Mistral) use eos as pad: with
    left padding and an attention mask, pad only fills the left side of shorter prompts (the scored last position is
    always a real token)."""
    from transformers import AutoTokenizer
    src, rev = source(name, model_path)
    tok = AutoTokenizer.from_pretrained(src, revision=rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(name, model_path=None):
    """bf16 model on one CUDA device, eval mode."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    src, rev = source(name, model_path)
    cls = AutoModelForCausalLM if MODELS[name][2] == "causal_lm" else AutoModelForImageTextToText
    return cls.from_pretrained(src, revision=rev, dtype=torch.bfloat16).to("cuda").eval()


def yes_no_ids(name, tok):
    """(Yes id, No id); each word must be a single token (and equal the expected ids where listed)."""
    ids = {w: tok.encode(w, add_special_tokens=False) for w in ("Yes", "No")}
    assert all(len(v) == 1 for v in ids.values()), ids
    y, n = ids["Yes"][0], ids["No"][0]
    if name in YES_NO_IDS:
        assert (y, n) == YES_NO_IDS[name], (name, y, n)
    return y, n


def messages(context, question):
    """Chat messages of the Yes/No judge prompt."""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{context}\n\n{question} Answer Yes or No."}]


def judge_prompt(tok, context, question):
    """The exact text fed to a local judge (tokenised afterwards with add_special_tokens=False)."""
    return tok.apply_chat_template(messages(context, question), tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def softmax(x, axis=-1):
    """Numerically stable softmax (numpy)."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
