"""All three datasets: cleaning, labels and one export row per kept state (question, history, window, scored calls).

Reads the per-dataset work files of steps 01-06, the FC-RewardBench source rows (tool schemas), the ComplexFuncBench
gold file, the FinTrace test set, the AgentSuite audit lists and the hand-made FC-RewardBench list of 35
system-prompt-only tasks (default: deref/build/config/fcrb_system_prompt_only_tasks.json). Writes
<work_dir>/states/export.jsonl and <work_dir>/states/cleaning.json. Needs `datasets`. CPU, a few minutes.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.build import export  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fcrb-raw", required=True, help="FC-RewardBench save_to_disk directory")
    ap.add_argument("--cfb-gold", required=True, help="ComplexFuncBench.jsonl")
    ap.add_argument("--testset", required=True, help="FinTrace evaluation/testset.json")
    ap.add_argument("--agentsuite-dir", required=True, help="directory with cfbench_issues.csv and cfbench_filter_ids.csv")
    ap.add_argument("--system-only", default=str(export.SYSTEM_ONLY), help="JSON list of FC-RewardBench test ids to drop")
    ap.add_argument("--work-dir", required=True, help="build work directory")
    a = ap.parse_args()
    w = Path(a.work_dir)
    export.export(w, Path(a.fcrb_raw), Path(a.cfb_gold), Path(a.testset), Path(a.agentsuite_dir), w / "states",
                  system_only=Path(a.system_only))


if __name__ == "__main__":
    main()
