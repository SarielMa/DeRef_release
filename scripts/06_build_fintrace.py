"""FinTrace: correct calls, constructed wrong calls (seed 20260930) and per-argument inputs.

Reads the test set and <work_dir>/fintrace/{tools.jsonl,tool_sim.npz}. Writes <work_dir>/fintrace/processed/
{positives,negatives}.jsonl, <work_dir>/fintrace/items/{items,calls}.jsonl and
<work_dir>/fintrace/items_nodesc/items.jsonl. CPU, a few minutes.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.build import fintrace  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", required=True, help="FinTrace evaluation/testset.json")
    ap.add_argument("--work-dir", required=True, help="build work directory (after 05_build_fintrace_tools.py)")
    a = ap.parse_args()
    w = Path(a.work_dir) / "fintrace"
    fintrace.build_samples(Path(a.testset), w / "tools.jsonl", w / "tool_sim.npz", w / "processed")
    fintrace.build_items(w / "processed", w / "tools.jsonl", w / "items")
    fintrace.build_items_nodesc(w / "items", w / "items_nodesc")


if __name__ == "__main__":
    main()
