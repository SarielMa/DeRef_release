"""Length guard before scoring: rebuild every text a judge will see and exit 1 if any input would be truncated.

Reads an items file (JSONL rows with "item_id", "context", "question"). Writes nothing.
  clm:          state_text(context, question) with the Qwen3-8B tokenizer (+ special tokens), limit 2,048;
                with --server (the running vLLM /v1/embeddings URL) the 20 longest inputs are also sent untruncated
                and the server's prompt_tokens must equal the local count.
  local judges: the chat prompt with the judge's own tokenizer, limit = max_position_embeddings.
(The Jev length check is part of scripts/45_score_jev.py, option --probe-max.)

  python scripts/40_check_lengths.py --judge clm --items <items.jsonl> [--server http://localhost:8090/v1/embeddings]
  python scripts/40_check_lengths.py --judge qwen3_8b --items <items.jsonl>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import read_jsonl  # noqa: E402
from deref.score import models as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", required=True, choices=["clm"] + sorted(M.MODELS))
    ap.add_argument("--items", required=True, help="judge-input JSONL (item_id, context, question)")
    ap.add_argument("--server", default=None, help="CLM only: vLLM embeddings URL for the server-side check")
    ap.add_argument("--model-path", default=None,
                    help="local model / tokenizer directory instead of the pinned HF id + revision")
    a = ap.parse_args()
    items = read_jsonl(a.items)
    if a.judge == "clm":
        from deref.score import clm_judge
        clm_judge.length_check(items, a.server, a.model_path)
        return
    from deref.score import local_judge
    n, limit = local_judge.length_report(items, a.judge, a.model_path)
    over = sum(v > limit for v in n)
    print(f"[length] {a.judge} ({a.items}): {len(n)} inputs, max {max(n)} tokens, total {sum(n):,}, limit {limit}, "
          f"over limit {over}")
    sys.exit(1 if over else 0)


if __name__ == "__main__":
    main()
