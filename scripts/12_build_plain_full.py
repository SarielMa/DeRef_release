"""Build the Plain-full judge inputs (complete history, no return cut, 32,768-token deletion ladder under the Qwen3-8B
and Qwen3.8-27B tokenizers).

Reads <data-dir>/data/{cfb,fintrace}.jsonl (FC-RewardBench has no history: its Plain-full input is the Plain input).
Writes <out>/plain_full_<dataset>.jsonl, rows {"item_id", "context", "question", "tokens", "cut", "level", "state"}
(level = [returns dropped, steps dropped]; item_id = argument item_id).

  python scripts/12_build_plain_full.py --data-dir <deref_data> --out <inputs_dir>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import write_jsonl  # noqa: E402
from deref.inputs import full_text as FT  # noqa: E402
from deref.inputs import state as ST  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--datasets", nargs="+", default=["cfb", "fintrace"], choices=ST.DATASETS)
    ap.add_argument("--all-arguments", action="store_true", help="also arguments outside the analysis set")
    ap.add_argument("--tokenizer-qwen3-8b", default=None, help="local tokenizer directory (default: Hugging Face id)")
    ap.add_argument("--tokenizer-qwen3-8-27b", default=None, help="local tokenizer directory (default: Hugging Face id)")
    a = ap.parse_args()
    b = FT.Builder(ST.load_tokenizer(ST.QWEN3_8B, a.tokenizer_qwen3_8b),
                   ST.load_tokenizer(ST.QWEN3_8_27B, a.tokenizer_qwen3_8_27b))
    for d in a.datasets:
        rows = FT.build_plain_full(ST.load_records(a.data_dir, d), b, analysis_only=not a.all_arguments)
        write_jsonl(Path(a.out) / f"plain_full_{d}.jsonl", rows)
        print(f"[plain_full] {d}: {len(rows)} items, {sum(r['cut'] for r in rows)} cut, "
              f"max {max((r['tokens'] for r in rows), default=0)} tokens")


if __name__ == "__main__":
    main()
