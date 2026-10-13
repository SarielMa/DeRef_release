"""Score full-history judge inputs (Plain-full, DeRef + full history; up to 32,768 tokens) with a local Yes/No judge.

Same prompt, system prompt, model loading and score as 42_score_local_judge.py; items are sorted by length and batched
under a padded-token budget, and only the last position's logits are computed (logits_to_keep=1). Output in input order.
Paper runs: qwen3_8b with --budget 65536, qwen3_8_27b with --budget 32768, --max-batch 32 (the defaults).
Reads an items file (JSONL rows with "item_id", "context", "question").
Writes <out>, rows {"item_id", "score", "yes_no_mass", "latency_ms", "tokens"}.

  python scripts/43_score_local_judge_long.py --judge qwen3_8_27b --items <items.jsonl> --out <scores>/cfb/plain_full/qwen3_8_27b.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import read_jsonl  # noqa: E402
from deref.score import models as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", required=True, choices=sorted(M.MODELS))
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True, help="output JSONL score file")
    ap.add_argument("--budget", type=int, default=None, help="max padded tokens per batch (default: as run)")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--model-path", default=None, help="local model directory instead of the pinned HF id + revision")
    a = ap.parse_args()
    from deref.score import local_judge
    t0 = time.time()
    print(f"[judge] system prompt: {M.SYSTEM!r}")
    items = read_jsonl(a.items)
    res = local_judge.score_items_long(items, a.judge, a.budget or M.MODELS[a.judge][4], a.max_batch, a.model_path)
    v = np.array([r["score"] for r in res])
    assert np.all(np.isfinite(v)) and v.min() >= 0 and v.max() <= 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in res)
    print(f"[done] {a.judge}: {len(res)} items, scores in [{v.min():.4f}, {v.max():.4f}], {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
