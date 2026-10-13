"""Table 3: ablation of deref with Qwen3.8-27B (argument level, all labelled arguments).

Reads the data package and the Qwen3.8-27B score files of deref, sgv_prior, deref_qwen3_8b, expected_value and
deref_nothink. d = AUROC(reference) - AUROC(variant) with the paired cluster
bootstrap CI. Writes `ablation.csv` and `ablation_x100.csv` (all three datasets; the paper reports ComplexFuncBench).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.stats import io, load, tables  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="data package directory (its data/ folder or the folder itself)")
    ap.add_argument("--scores-dir", required=True, help="score files <scores_dir>/<dataset>/<input>/<judge>.jsonl")
    ap.add_argument("--out-dir", required=True, help="output directory for the CSV files")
    a = ap.parse_args()
    sets = load.load_sets(a.data_dir)
    inputs = ("plain",) + tuple(dict.fromkeys(x for ab in tables.ABL for x in ab[1:3]))
    SC, _ = load.load_scores(a.scores_dir, inputs, judges=(tables.ABL_JUDGE,))
    io.write_table(tables.ablation(sets, SC), a.out_dir, "ablation")


if __name__ == "__main__":
    main()
