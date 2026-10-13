"""FCRB-Relocate step 2 (GPU): rewrite the requests with Qwen3.8-27B (thinking off, greedy, 400 new tokens; vLLM,
bf16, engine seed 0, max_model_len = longest prompt + 416).

Reads the prompt file of step 1. Writes rows {"prompt_id", "output", "finished", "new_tokens"}.

  python scripts/61_relocate_rewrite_generate.py --prompts <relocate_dir>/rewrite_prompts.jsonl --out <relocate_dir>/rewrite_generations.jsonl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.build import relocate as RL  # noqa: E402
from deref.common import read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    RL.generate_rewrites(read_jsonl(a.prompts), a.out)


if __name__ == "__main__":
    main()
