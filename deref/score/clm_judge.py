"""CLM judge: Qwen3-8B embeddings from a vLLM pooling server (last-token pooling) + the CLM projection heads
(`clm` package, contrastive-lm 0.1.0, https://github.com/Contrastive-LM/CLM; head checkpoint CLM_v0.1-8B.pt from
the Hugging Face repo Contrastive-LM/CLM-v0.1-8B, sha256 b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5).

Server (one GPU), see scripts/41_score_clm.sh:
  vllm serve Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --served-model-name qwen3-8b \
    --runner pooling --pooler-config '{"seq_pooling_type": "LAST"}' --max-model-len 2048 --dtype bfloat16 \
    --gpu-memory-utilization 0.5 --port 8090
  The Table 1 Plain run used --max-model-len 2048 (the default). All later CLM runs used --max-model-len 2064: with
  concurrent requests vLLM can hang on an input of exactly max-model-len tokens, and 2064 avoids that. The client
  truncates at 2,048 tokens in both cases and every input is <= 2,048 tokens, so the setting changes no input.

Per item (no chat template, no system prompt):
  state text  = clm.schema.state_text(context, question) = context.strip() + "\n\n" + question.strip()
  candidates  = clm.schema.candidates({"type": "noul", "instructions": question})
              = ["false: No. This is false: {question}", "true: Yes. This is true: {question}"]
  score       = softmax(scale * state_head(e_state) @ action_head(e_candidates).T)[true]
Self-check: the first 30 items are re-answered with clm.Engine.answer; max abs diff must be < 1e-3.
"""
import os
import sys
import time

from deref.score import models as M

CLM_LIMIT = 2048
DEFAULT_EMB_URL = "http://localhost:8090/v1/embeddings"
SERVED_NAME = "qwen3-8b"


def score_items(items, checkpoint, emb_url=DEFAULT_EMB_URL):
    """Rows {"item_id", "score", "latency_ms"}, grouped by question (sorted), 256 items per embedding request group."""
    from clm import Embedder, Engine
    from clm.heads import HeadPair
    from clm.schema import candidates, state_text
    emb = Embedder(url=emb_url, model=SERVED_NAME, max_tokens=CLM_LIMIT, batch=64)
    head = HeadPair("clm", checkpoint).ensure()
    rows = []
    for q in sorted({x["question"] for x in items}):
        sub = [x for x in items if x["question"] == q]
        keys, texts = candidates({"type": "noul", "instructions": q})
        e, _ = emb.embed(texts)
        za = head.project_actions(e).float().cpu().numpy()
        t = keys.index("true")
        for i in range(0, len(sub), 256):
            ch = sub[i:i + 256]
            tb = time.time()
            e, _ = emb.embed([state_text(x["context"], q) for x in ch])
            zq = head.project_states(e).float().cpu().numpy()
            p = M.softmax(head.scale * zq @ za.T)[:, t]
            ms = (time.time() - tb) * 1000 / len(ch)
            rows += [{"item_id": x["item_id"], "score": float(v), "latency_ms": ms} for x, v in zip(ch, p)]
    eng = Engine(embedder=emb, checkpoint=checkpoint, action_cache="0")
    byid = {r["item_id"]: r["score"] for r in rows}
    diffs = [abs(eng.answer(x["context"], {"q": {"type": "noul", "instructions": x["question"]}})["answers"]["q"]["noul"]
                 - byid[x["item_id"]]) for x in items[:30]]
    print(f"[check] CLM batched vs Engine.answer on {len(diffs)} items: max abs diff {max(diffs):.2e}")
    assert max(diffs) < 1e-3
    return rows


def length_check(items, server=None, tokenizer_path=None):
    """Exit 1 if any CLM input exceeds 2,048 tokens (Qwen3-8B tokenizer, special tokens added as vLLM adds them).
    With `server` (the vLLM /v1/embeddings URL) the 20 longest inputs are also sent WITHOUT truncation: the server
    must accept them and its prompt_tokens must equal the local count."""
    import requests
    from clm.schema import state_text
    from transformers import AutoTokenizer
    src, rev = (tokenizer_path, None) if tokenizer_path else M.QWEN3_8B
    tok = AutoTokenizer.from_pretrained(src, revision=rev)
    texts = [state_text(x["context"], x["question"]) for x in items]
    n = [len(i) for i in tok(texts, add_special_tokens=True)["input_ids"]]
    over = sum(x > CLM_LIMIT for x in n)
    print(f"[length] clm: {len(n)} inputs, max {max(n)} tokens, limit {CLM_LIMIT}, over limit {over}")
    if server:
        top = sorted(zip(n, texts), reverse=True)[:20]
        r = requests.post(server, json={"model": SERVED_NAME, "input": [t for _, t in top], "encoding_format": "float"},
                          timeout=900)
        if r.status_code != 200 or int(r.json()["usage"]["prompt_tokens"]) != sum(x for x, _ in top):
            print(f"[length] FAIL: server check {r.status_code} {r.text[:200]}")
            sys.exit(1)
        print("[length] server check: 20 longest inputs accepted untruncated, token counts match")
    if over:
        print(f"[length] FAIL: {over} inputs exceed the limit and would be truncated")
        sys.exit(1)
    return n


def checkpoint_from_env(arg):
    """Head checkpoint path: the argument, else env CLM_CKPT."""
    ck = arg or os.environ.get("CLM_CKPT")
    if not ck:
        sys.exit("give --checkpoint or set CLM_CKPT to the CLM_v0.1-8B.pt head checkpoint")
    return ck
