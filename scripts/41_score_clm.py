"""Score judge-input items with CLM (needs the vLLM pooling server for Qwen3-8B; see scripts/41_score_clm.sh).

Reads an items file (JSONL rows with "item_id", "context", "question") and the CLM head checkpoint.
Writes <out> (e.g. <scores_dir>/<dataset>/<input>/clm.jsonl), rows {"item_id", "score", "latency_ms"};
score = CLM's probability of "true" for the Noul question (the item's question) on the state context + question.
Run 40_check_lengths.py --judge clm first (inputs must be <= 2,048 tokens).

  python scripts/41_score_clm.py --items <items.jsonl> --out <scores>/cfb/plain/clm.jsonl \
      --checkpoint <CLM_v0.1-8B.pt> [--emb-url http://localhost:8090/v1/embeddings]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import read_jsonl  # noqa: E402
from deref.score import clm_judge  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True, help="output JSONL score file")
    ap.add_argument("--checkpoint", default=None, help="CLM head checkpoint (default: env CLM_CKPT)")
    ap.add_argument("--emb-url", default=os.environ.get("CLM_EMB_URL", clm_judge.DEFAULT_EMB_URL),
                    help="vLLM /v1/embeddings URL (default: env CLM_EMB_URL or %(default)s)")
    a = ap.parse_args()
    items = read_jsonl(a.items)
    rows = clm_judge.score_items(items, clm_judge.checkpoint_from_env(a.checkpoint), a.emb_url)
    v = np.array([r["score"] for r in rows])
    assert len(rows) == len(items) and np.all(np.isfinite(v)) and v.min() >= 0 and v.max() <= 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"[done] clm: {len(rows)} items, scores in [{v.min():.4f}, {v.max():.4f}]")


if __name__ == "__main__":
    main()
