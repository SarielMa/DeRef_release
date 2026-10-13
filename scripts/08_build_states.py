"""Final states with the 2,048-token rule (Qwen/Qwen3-8B tokenizer @ b968826d..., CPU) and argument labels.

Reads <work_dir>/states/export.jsonl. Writes <work_dir>/states/<dataset>/{positives,negatives,calls,excluded,labels}.jsonl
(positives carry the final state and its ladder rung). The datasets can be run as separate processes
(`--datasets fintrace`). CPU: FC-RewardBench and ComplexFuncBench a few minutes, FinTrace about an hour.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.build import states  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True, help="build work directory (after 07_export_states.py)")
    ap.add_argument("--datasets", nargs="+", choices=("fcrb", "cfb", "fintrace"), help="only these datasets (default: all)")
    a = ap.parse_args()
    w = Path(a.work_dir) / "states"
    states.build_states(w / "export.jsonl", w, datasets=a.datasets)


if __name__ == "__main__":
    main()
