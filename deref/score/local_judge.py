"""Local Yes/No judges (Qwen3 family, Qwen3.8-27B, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3) with plain
Hugging Face transformers: one forward pass per prompt, no generation, no sampling.

score = softmax(logits[last position, [Yes, No]])[Yes]; `yes_no_mass` = full-softmax P(Yes) + P(No) is kept as a check.

- `score_items`:      fixed-size batches in input order (all truncated inputs: Plain, value search, reference reading).
- `score_items_long`: full-history inputs (up to 32,768 tokens): items sorted by length (longest first) and batched
                      under a padded-token budget; only the last position's logits are computed (`logits_to_keep=1`).
                      Same prompt and maths; output in input order.
- `length_report`:    rebuilds every prompt, counts tokens with the judge's own tokenizer and compares with the model's
                      max_position_embeddings (inputs are never truncated; this only guards against overflow).
"""
import time

import numpy as np

from deref.score import models as M


def score_items(items, name, batch, model_path=None):
    """Rows {"item_id", "score", "yes_no_mass", "latency_ms"} in input order."""
    import torch
    tok = M.load_tokenizer(name, model_path)
    model = M.load_model(name, model_path)
    y, n = M.yes_no_ids(name, tok)

    def prompt(x):
        return M.judge_prompt(tok, x["context"], x["question"])
    print(f"[judge] {name} example prompt:\n{prompt(items[0])}")
    rows = []
    for i in range(0, len(items), batch):
        ch = items[i:i + batch]
        tb = time.time()
        enc = tok([prompt(x) for x in ch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            lg = model(**enc).logits[:, -1, :].float()
        full = torch.softmax(lg, -1)
        two = torch.softmax(lg[:, [y, n]], -1)[:, 0]
        mass = full[:, y] + full[:, n]
        ms = (time.time() - tb) * 1000 / len(ch)
        rows += [{"item_id": x["item_id"], "score": float(p), "yes_no_mass": float(m), "latency_ms": ms}
                 for x, p, m in zip(ch, two.cpu().numpy(), mass.cpu().numpy())]
    m = np.array([r["yes_no_mass"] for r in rows])
    print(f"[check] {name}: Yes+No mass min {m.min():.3f}, p5 {np.percentile(m, 5):.3f}, median {np.median(m):.3f}")
    return rows


def length_batches(lens, budget, max_batch):
    """Index batches: longest first; a batch closes when (size + 1) * longest > budget or size == max_batch."""
    order = sorted(range(len(lens)), key=lambda i: -lens[i])
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lens[cur[0]] > budget or len(cur) == max_batch:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def score_items_long(items, name, budget, max_batch=32, model_path=None):
    """Rows {"item_id", "score", "yes_no_mass", "latency_ms", "tokens"} in input order."""
    import torch
    tok = M.load_tokenizer(name, model_path)
    model = M.load_model(name, model_path)
    y, n = M.yes_no_ids(name, tok)
    texts = [M.judge_prompt(tok, x["context"], x["question"]) for x in items]
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    print(f"[length] {name}: {len(items)} inputs, max {max(lens)} tokens, median {int(np.median(lens))}")
    batches = length_batches(lens, budget, max_batch)
    res = [None] * len(items)
    t0 = time.time()
    for bi, b in enumerate(batches):
        tb = time.time()
        enc = tok([texts[i] for i in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            lg = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        full = torch.softmax(lg, -1)
        two = torch.softmax(lg[:, [y, n]], -1)[:, 0].cpu().numpy()
        mass = (full[:, y] + full[:, n]).cpu().numpy()
        ms = (time.time() - tb) * 1000 / len(b)
        for i, p, m in zip(b, two, mass):
            res[i] = {"item_id": items[i]["item_id"], "score": float(p), "yes_no_mass": float(m), "latency_ms": ms,
                      "tokens": lens[i]}
        if bi % 50 == 0:
            print(f"[progress] batch {bi + 1}/{len(batches)} ({len(b)} items, {lens[b[0]]} tokens) "
                  f"{time.time() - t0:.0f}s", flush=True)
    m = np.array([r["yes_no_mass"] for r in res])
    print(f"[check] {name}: Yes+No mass min {m.min():.3f}, p5 {np.percentile(m, 5):.3f}, median {np.median(m):.3f}")
    return res


def length_report(items, name, model_path=None):
    """(token counts, limit) for the judge prompts of `items`; limit = max_position_embeddings."""
    from transformers import AutoConfig
    tok = M.load_tokenizer(name, model_path)
    src, rev = M.source(name, model_path)
    cfg = AutoConfig.from_pretrained(src, revision=rev)
    limit = getattr(cfg, "max_position_embeddings", None) or cfg.text_config.max_position_embeddings
    n = [len(tok(M.judge_prompt(tok, x["context"], x["question"]), add_special_tokens=False)["input_ids"])
         for x in items]
    return n, limit
