"""Generate the self-generated references with the judge model itself (GPU, vLLM 0.30.0, bf16, one GPU).

Greedy decoding, at most 384 new tokens, thinking off, engine and request seed 20260930, max_model_len 8,192 (as run
for the paper: generation took about 1 min (Qwen3-8B, B200) and 8 min (Qwen3.8-27B, H200) for the 10,741 prompts
of the three datasets, plus model loading).
Reads the prompt file of 80_build_self_reference_prompts.py. Writes rows {"prompt_id", "output", "new_tokens",
"prompt_tokens", "finish_reason", "empty"}.

  python scripts/81_generate_self_reference.py --judge qwen3_8b \\
      --prompts <gen_dir>/prompts/self_reference_qwen3_8b.jsonl --out <gen_dir>/generations/self_reference_qwen3_8b.jsonl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.generate import self_reference as SR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", required=True, choices=sorted(SR.JUDGE_MODEL))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    SR.generate(a.prompts, a.out, a.judge)


if __name__ == "__main__":
    main()
