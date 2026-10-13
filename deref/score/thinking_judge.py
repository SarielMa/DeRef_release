"""Qwen3.8-27B Yes/No judge with thinking on (Table 4), in two stages.

Prompt: the local-judge system / user messages of `deref.score.models` rendered with enable_thinking=True and
reasoning_effort="medium" (the text ends with "<|im_start|>assistant\\n<think>\\n"; for "medium" the template adds nothing
to the system message).
Stage 1 (vLLM): sample the thinking until `</think>` with the model's generation settings (temperature 1.0, top_p 0.95,
  top_k 20), request seed 20260930, at most CAP = 8,192 new tokens, max_model_len 40,960, engine seed 0.
Stage 2 (transformers, bf16): prompt + thinking tokens (end tokens removed and `</think>` appended when the model did not
  close the thinking) + "\\n\\n"; score = P(Yes) / (P(Yes) + P(No)) at the next position. Batches are cut by a padded-token
  budget (rows sorted by length, at most `batch` rows, rows * longest <= `budget`), left padding, last-position logits only.
The thinking-off render of the same item is the local judge prompt; on and off differ only in the final tail
("<think>\\n" vs "<think>\\n\\n</think>\\n\\n").
"""
import json
import time
from pathlib import Path

import numpy as np

from deref.score import models as M

JUDGE = "qwen3_8_27b"
THINK_END, NL2, YES, NO = 248069, 271, 9175, 2665  # `</think>`, "\n\n", "Yes", "No" in the Qwen3.8-27B tokenizer
EOS = (248046, 248044)
CAP, CTX, SEED = 8192, 40960, 20260930
OFF_TAIL, ON_TAIL = "<think>\n\n</think>\n\n", "<think>\n"


def load_tokenizer(model_path=None):
    from transformers import AutoTokenizer
    src, rev = M.source(JUDGE, model_path)
    tok = AutoTokenizer.from_pretrained(src, revision=rev)
    assert tok.encode("</think>\n\n", add_special_tokens=False) == [THINK_END, NL2]
    assert tok.encode("Yes", add_special_tokens=False) == [YES] and tok.encode("No", add_special_tokens=False) == [NO]
    return tok


def prompt(tok, x):
    """Thinking-on judge prompt of one item."""
    t = tok.apply_chat_template(M.messages(x["context"], x["question"]), tokenize=False, add_generation_prompt=True,
                                enable_thinking=True, reasoning_effort="medium")
    assert t.endswith("<|im_start|>assistant\n<think>\n"), t[-80:]
    return t


def check_render(tok, x):
    """True if the thinking-on prompt equals the thinking-off judge prompt up to the final tail."""
    on, off = prompt(tok, x), M.judge_prompt(tok, x["context"], x["question"])
    return on.endswith(ON_TAIL) and off.endswith(OFF_TAIL) and on[:-len(ON_TAIL)] == off[:-len(OFF_TAIL)]


def dedup(files):
    """Distinct (context, question) inputs over several items files: (unique rows, {file name: {item_id: unique id}})."""
    uniq, mp = {}, {}
    for f in files:
        m = {}
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                k = (r["context"], r["question"])
                if k not in uniq:
                    uniq[k] = {"item_id": f"u{len(uniq):05d}", "context": r["context"], "question": r["question"]}
                assert r["item_id"] not in m
                m[r["item_id"]] = uniq[k]["item_id"]
        assert Path(f).name not in mp, f"two items files named {Path(f).name}"
        mp[Path(f).name] = m
    return list(uniq.values()), mp


def generate(items, out_path, model_path=None):
    """Stage 1: one thinking sequence per item -> rows {"item_id", "prompt_tokens", "gen_ids", "closed",
    "finish_reason", "stop_reason", "think_tokens"}."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = load_tokenizer(model_path)
    ids = [tok(prompt(tok, x), add_special_tokens=False)["input_ids"] for x in items]
    assert max(map(len, ids)) + CAP + 2 <= CTX
    print(f"[think-gen] {len(items)} items, {sum(map(len, ids)):,} prompt tokens, max {max(map(len, ids))}", flush=True)
    src, rev = M.source(JUDGE, model_path)
    llm = LLM(model=src, revision=rev, dtype="bfloat16", max_model_len=CTX, gpu_memory_utilization=0.9,
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=CAP, seed=SEED, stop_token_ids=[THINK_END],
                        skip_special_tokens=False)
    t0 = time.time()
    res = llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], sp)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n_cap = 0
    with open(out_path, "w") as f:
        for x, p, r in zip(items, ids, res):
            o = r.outputs[0]
            g = list(o.token_ids)
            closed = bool(g) and g[-1] == THINK_END
            if not closed and o.finish_reason == "stop" and o.stop_reason == THINK_END:  # stop token not kept in token_ids
                g, closed = g + [THINK_END], True
            n_cap += o.finish_reason == "length"
            f.write(json.dumps({"item_id": x["item_id"], "prompt_tokens": len(p), "gen_ids": g, "closed": closed,
                                "finish_reason": o.finish_reason, "stop_reason": o.stop_reason,
                                "think_tokens": len(g)}) + "\n")
    print(f"[done] {len(items)} items in {time.time() - t0:.0f} s; hit the {CAP}-token cap: {n_cap}", flush=True)


def score(items, think, batch=8, budget=40000, model_path=None):
    """Stage 2: rows {"item_id", "score", "yes_no_mass", "think_tokens", "closed", "finish_reason", "prompt_tokens"}
    in input order. `think` = {item_id: stage-1 row}."""
    import inspect

    import torch
    from transformers import AutoModelForImageTextToText
    assert set(think) >= {x["item_id"] for x in items}
    tok = load_tokenizer(model_path)
    src, rev = M.source(JUDGE, model_path)
    model = AutoModelForImageTextToText.from_pretrained(src, revision=rev, dtype=torch.bfloat16).to("cuda").eval()
    keep = "logits_to_keep" in inspect.signature(model.forward).parameters
    seqs, st = [], {"closed by the model": 0, "</think> appended (cap or end token)": 0}
    for x in items:
        r = think[x["item_id"]]
        p = tok(prompt(tok, x), add_special_tokens=False)["input_ids"]
        assert len(p) == r["prompt_tokens"]
        g = list(r["gen_ids"])
        if r["closed"]:
            st["closed by the model"] += 1
        else:
            g = [t for t in g if t not in EOS] + [THINK_END]
            st["</think> appended (cap or end token)"] += 1
        seqs.append(p + g + [NL2])
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    batches, cur = [], []
    for k in order:  # ascending length: padded size of a batch = len(last) * rows
        if cur and (len(cur) >= batch or len(seqs[k]) * (len(cur) + 1) > budget):
            batches.append(cur)
            cur = []
        cur.append(k)
    batches.append(cur)
    out = [None] * len(seqs)
    pad = tok.pad_token_id
    t0 = time.time()
    for bi, b in enumerate(batches):
        L = max(len(seqs[k]) for k in b)
        ids = torch.full((len(b), L), pad, dtype=torch.long)
        am = torch.zeros((len(b), L), dtype=torch.long)
        for r_, k in enumerate(b):  # left padding
            s = seqs[k]
            ids[r_, L - len(s):] = torch.tensor(s)
            am[r_, L - len(s):] = 1
        with torch.no_grad():
            kw = {"logits_to_keep": 1} if keep else {}
            lg = model(input_ids=ids.cuda(), attention_mask=am.cuda(), **kw).logits[:, -1, :].float()
        full = torch.softmax(lg, -1)
        two = torch.softmax(lg[:, [YES, NO]], -1)[:, 0]
        mass = full[:, YES] + full[:, NO]
        for r_, k in enumerate(b):
            r = think[items[k]["item_id"]]
            out[k] = {"item_id": items[k]["item_id"], "score": float(two[r_]), "yes_no_mass": float(mass[r_]),
                      "think_tokens": r["think_tokens"], "closed": r["closed"], "finish_reason": r["finish_reason"],
                      "prompt_tokens": r["prompt_tokens"]}
        if bi % 200 == 0:
            print(f"[score] batch {bi}/{len(batches)} (L {L}, rows {len(b)}), {time.time() - t0:.0f} s", flush=True)
    m = np.array([r["yes_no_mass"] for r in out])
    print(f"[check] Yes+No mass min {m.min():.3f}, p5 {np.percentile(m, 5):.3f}, median {np.median(m):.3f}; {st}",
          flush=True)
    return out
