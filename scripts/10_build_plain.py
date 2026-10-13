"""Build the Plain judge inputs (one per argument) from the data package.

Reads <data-dir>/data/{fcrb,cfb,fintrace}.jsonl. Writes <out>/plain_<dataset>.jsonl, rows {"item_id", "context",
"question"} (item_id = argument item_id). No tokenizer is needed: the 2,048-token rung of each state is stored in the
record (`plain_rung`).

  python scripts/10_build_plain.py --data-dir <deref_data> --out <inputs_dir>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import write_jsonl  # noqa: E402
from deref.inputs import plain  # noqa: E402
from deref.inputs import state as ST  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--datasets", nargs="+", default=list(ST.DATASETS), choices=ST.DATASETS)
    ap.add_argument("--all-arguments", action="store_true", help="also arguments outside the analysis set")
    a = ap.parse_args()
    for d in a.datasets:
        rows = plain.build(ST.load_records(a.data_dir, d), analysis_only=not a.all_arguments)
        write_jsonl(Path(a.out) / f"plain_{d}.jsonl", rows)
        print(f"[plain] {d}: {len(rows)} items")


if __name__ == "__main__":
    main()
