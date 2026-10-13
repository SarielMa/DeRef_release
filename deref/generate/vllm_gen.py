"""Offline vLLM generation over a prompt file (GPU).

Prompts are the chat-templated texts of `deref.generate.prompts` (rows {"prompt_id", "text", "tokens", ...}), fed as
token ids (tokenizer of the model, no special tokens added). The stored token count is re-asserted.

Two decoding modes, exactly as run for the paper (vLLM 0.30.0, bf16, one GPU, engine seed 0, gpu_memory_utilization 0.9):

* greedy (thinking-off originals): temperature 0, `max_new` new tokens (default 160; 1,024 for the derivation-style
  prompts). Qwen3.8-27B: max_model_len = 32,768 + max_new, prompt <= 32,768 tokens. Qwen3-8B: max_model_len 40,960,
  prompt <= 40,960 - max_new tokens.
* thinking (paper methods): sampling with the model's generation_config values (Qwen3.8-27B temperature 1.0, Qwen3-8B
  temperature 0.6; both top_p 0.95, top_k 20, min_p not set) and one SamplingParams seed 20260930 shared by every
  request; max_new 8,192; max_model_len 40,960; prompt <= 32,768 tokens. Records the index of the last `</think>`
  token in the output (`think_end_index`, None if absent). The thinking text, `</think>` and the answer are all in
  `output` (the opening `<think>` is part of the prompt).

Output rows: {"prompt_id", "output", "new_tokens", "finished", "finish_reason", "prompt_tokens"} (+ "think_end_index"
in thinking mode). Byte-identical regeneration is not guaranteed (batch-dependent kernels under continuous batching).
"""
import json
import time
from pathlib import Path

from deref.generate.models import MODELS

LOC_LIMIT = 32768
CTX = 40960
SEED = 20260930
TEMPERATURE = {"qwen3_8_27b": 1.0, "qwen3_8b": 0.6}


def _load(prompts_path, model):
    from transformers import AutoTokenizer
    m = MODELS[model]
    with open(prompts_path, encoding="utf-8") as f:
        prompts = [json.loads(l) for l in f]
    tok = AutoTokenizer.from_pretrained(m["repo_id"], revision=m["revision"])
    ids = [tok(p["text"], add_special_tokens=False)["input_ids"] for p in prompts]
    return prompts, ids


def _llm(model, max_model_len):
    from vllm import LLM
    m = MODELS[model]
    return LLM(model=m["repo_id"], revision=m["revision"], tokenizer_revision=m["revision"], dtype="bfloat16",
               max_model_len=max_model_len, gpu_memory_utilization=0.9, limit_mm_per_prompt={"image": 0, "video": 0},
               seed=0)


def generate(prompts_path, out_path, model="qwen3_8_27b", mode="greedy", max_new=160):
    """Generate one output per prompt and write the output rows to `out_path`."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    prompts, ids = _load(prompts_path, model)
    if mode == "greedy":
        if model == "qwen3_8b":
            limit, max_model_len = CTX - max_new, CTX
        else:
            limit, max_model_len = LOC_LIMIT, LOC_LIMIT + max_new
    elif mode == "thinking":
        limit, max_model_len = LOC_LIMIT, CTX
        assert LOC_LIMIT + max_new <= CTX
    else:
        raise ValueError(mode)
    for p, x in zip(prompts, ids):
        assert len(x) == p["tokens"] <= limit, p["prompt_id"]
    print(f"[gen] {len(prompts)} prompts, {sum(map(len, ids)):,} prompt tokens, max {max(map(len, ids))}")
    print(f"[gen] example prompt:\n{prompts[0]['text']}")
    t0 = time.time()
    llm = _llm(model, max_model_len)
    print(f"[gen] model loaded in {time.time() - t0:.0f} s")
    if mode == "greedy":
        sp = SamplingParams(temperature=0.0, max_tokens=max_new, skip_special_tokens=True)
    else:
        sp = SamplingParams(temperature=TEMPERATURE[model], top_p=0.95, top_k=20, max_tokens=max_new, seed=SEED,
                            skip_special_tokens=True)
    think_end = MODELS[model]["think_end"]
    t1 = time.time()
    res = llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], sp)
    dt = time.time() - t1
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    unf = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for p, x, r in zip(prompts, ids, res):
            o = r.outputs[0]
            fin = o.finish_reason == "stop"
            unf += not fin
            row = {"prompt_id": p["prompt_id"], "output": o.text, "new_tokens": len(o.token_ids),
                   "finished": fin, "finish_reason": o.finish_reason, "prompt_tokens": len(x)}
            if mode == "thinking":
                tid = list(o.token_ids)
                row["think_end_index"] = len(tid) - 1 - tid[::-1].index(think_end) if think_end in tid else None
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[done] {len(prompts)} prompts in {dt:.0f} s (generation only); prompt tokens {sum(map(len, ids)):,} "
          f"({sum(map(len, ids)) / dt:,.0f} tokens/s); new tokens {sum(len(r.outputs[0].token_ids) for r in res):,}; "
          f"not finished within {max_new} tokens: {unf}")
