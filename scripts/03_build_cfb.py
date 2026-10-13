"""ComplexFuncBench: first-divergence steps, main sample, correct / wrong step pairs (length rule) and per-argument inputs.

Reads the gold ComplexFuncBench.jsonl (HF `zai-org/ComplexFuncBench`), the AgentSuite trajectories directory
(HF `AgentSuite/complex-func-bench-trajectories` @ fcc906da..., one <model>.jsonl per model) and the AgentSuite audit
lists (cfbench_filter_ids.csv, cfbench_issues.csv). Writes under <work_dir>/cfb/: first_divergence/pairs.jsonl,
main_sample/pairs.jsonl, processed/{positives,negatives,excluded_too_long,excluded_value_lost}.jsonl,
items/{items,calls}.jsonl. The length rule uses the Qwen/Qwen3-8B tokenizer (revision b968826d..., CPU).
About 5-10 minutes on CPU.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.build import cfb  # noqa: E402

STEPS = ("first_divergence", "main_sample", "samples", "items")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="ComplexFuncBench.jsonl")
    ap.add_argument("--trajectories", required=True, help="directory with the AgentSuite <model>.jsonl files")
    ap.add_argument("--agentsuite-dir", required=True, help="directory with cfbench_filter_ids.csv and cfbench_issues.csv")
    ap.add_argument("--work-dir", required=True, help="build work directory")
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS), help="run only these steps (default: all)")
    a = ap.parse_args()
    w = Path(a.work_dir) / "cfb"
    gold = Path(a.gold)
    if "first_divergence" in a.steps:
        cfb.first_divergence(gold, Path(a.trajectories), Path(a.agentsuite_dir), w / "first_divergence" / "pairs.jsonl")
    if "main_sample" in a.steps:
        cfb.main_sample(w / "first_divergence" / "pairs.jsonl", w / "main_sample" / "pairs.jsonl")
    if "samples" in a.steps:
        cfb.build_samples(gold, w / "main_sample" / "pairs.jsonl", w / "processed")
    if "items" in a.steps:
        cfb.build_items(gold, w / "processed", w / "items")


if __name__ == "__main__":
    main()
