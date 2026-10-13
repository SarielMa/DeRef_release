"""Score judge-input items with a local Yes/No judge (Hugging Face transformers, one GPU, bf16).

Judges: qwen3_4b, qwen3_8b, qwen3_8_27b (Qwen3.8-27B), llama3_1_8b, mistral_7b
(HF id + pinned revision in deref/score/models.py; --model-path loads a local snapshot instead).
Reads an items file (JSONL rows with "item_id", "context", "question").
Writes <out> (e.g. <scores_dir>/<dataset>/<input>/<judge>.jsonl), rows {"item_id", "score", "yes_no_mass", "latency_ms"};
score = P(Yes) / (P(Yes) + P(No)) at the first answer token of the prompt
  system "You are a careful evaluator of tool calls made by an AI assistant."
  user   "{context}\n\n{question} Answer Yes or No."   (chat template, enable_thinking=False, left padding).
Batch size as run: 32. Scores depend slightly on batch composition (padding), so keep it.
Run 40_check_lengths.py first. For full-history inputs use 43_score_local_judge_long.py.

  python scripts/42_score_local_judge.py --judge qwen3_8b --items <items.jsonl> --out <scores>/cfb/plain/qwen3_8b.jsonl
"""
import argparse
import json
import sys
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
    ap.add_argument("--batch", type=int, default=None, help="batch size (default: the size used in the paper runs)")
    ap.add_argument("--model-path", default=None, help="local model directory instead of the pinned HF id + revision")
    a = ap.parse_args()
    from deref.score import local_judge
    items = read_jsonl(a.items)
    print(f"[judge] system prompt: {M.SYSTEM!r}; model: {M.source(a.judge, a.model_path)}")
    rows = local_judge.score_items(items, a.judge, a.batch or M.MODELS[a.judge][3], a.model_path)
    v = np.array([r["score"] for r in rows])
    assert len(rows) == len(items) and np.all(np.isfinite(v)) and v.min() >= 0 and v.max() <= 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"[done] {a.judge}: {len(rows)} items, scores in [{v.min():.4f}, {v.max():.4f}]")


if __name__ == "__main__":
    main()
