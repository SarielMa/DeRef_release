"""Jev final step: every item takes the cached answer of the input it was sent with (its own input, or for refused
full-history items the ladder input k steps down) and the per-item score files are written.

Reads <work>/batch.json, the round files, k.json, the answer cache (read-only) and, only if some item needed the
ladder, the data package. Writes each batch's output score file, rows {"item_id", "score", "model", "k", "input_key"},
<work>/jev_in/batch<j>.jsonl (the inputs Jev read) and <work>/final_summary.json (checks).

  python scripts/47_jev_final.py --work <work> --cache <jev_cache.sqlite> [--data-dir <deref_data>]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.score import jev_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--data-dir", default=None, help="data package directory (needed when the ladder was used)")
    a = ap.parse_args()
    jev_batch.cmd_final(a.work, a.cache, a.data_dir)


if __name__ == "__main__":
    main()
