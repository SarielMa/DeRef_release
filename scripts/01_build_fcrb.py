"""FC-RewardBench: correct / wrong call pairs and per-argument inputs from the source rows.

Reads the `save_to_disk` copy of HF `ibm-research/fc-reward-bench` (revision 269929c3...). Writes
<work_dir>/fcrb/processed/{positives,negatives}.jsonl, <work_dir>/fcrb/items/{items,calls}.jsonl and
<work_dir>/fcrb/items_nodesc/items.jsonl. Needs the `datasets` package. CPU, about a minute.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.build import fcrb  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, help="directory written by datasets.save_to_disk (data-00000-of-00001.arrow, ...)")
    ap.add_argument("--work-dir", required=True, help="build work directory")
    a = ap.parse_args()
    w = Path(a.work_dir) / "fcrb"
    fcrb.build_samples(Path(a.raw_dir), w / "processed")
    fcrb.build_items(Path(a.raw_dir), w / "processed", w / "items")
    fcrb.build_items_nodesc(w / "items", w / "items_nodesc")


if __name__ == "__main__":
    main()
