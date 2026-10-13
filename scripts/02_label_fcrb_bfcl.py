"""FC-RewardBench argument labels with the official BFCL v3 AST checker.

Reads BFCL v3 questions / possible answers (HF `gorilla-llm/Berkeley-Function-Calling-Leaderboard` @ 61fc0608...:
<bfcl_dir>/questions/BFCL_v3_*.json, <bfcl_dir>/possible_answer/BFCL_v3_*.json) and imports the checker from a local
gorilla checkout at commit 70b6a4a21445 (`--gorilla-dir`: the checkout root or its berkeley-function-call-leaderboard
directory; the checker is not shipped here). Writes <work_dir>/fcrb/bfcl_param_labels.jsonl. CPU, seconds.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.build import fcrb  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfcl-dir", required=True, help="BFCL v3 data with questions/ and possible_answer/")
    ap.add_argument("--gorilla-dir", required=True, help="gorilla checkout (commit 70b6a4a21445)")
    ap.add_argument("--work-dir", required=True, help="build work directory (after 01_build_fcrb.py)")
    a = ap.parse_args()
    w = Path(a.work_dir) / "fcrb"
    fcrb.bfcl_labels(Path(a.bfcl_dir), a.gorilla_dir, w / "processed", w / "items", w / "bfcl_param_labels.jsonl")


if __name__ == "__main__":
    main()
