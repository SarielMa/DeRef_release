"""Table 2: argument AUROC averaged over the judges (Avg.) and the mean difference from deref (Delta) with its CI.

Reads the data package and the score files of all grid inputs. For each dataset and comparison (every grid input minus
deref, and deref minus plain) the judges' paired bootstrap differences are averaged draw by draw; the CI is the 2.5 / 97.5
percentile of the averaged draws. Judge sets: all four judges ("all") and without Jev ("without_jev").
Writes `judge_averages.csv` and `judge_averages_x100.csv`.
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
    SC, _ = load.load_scores(a.scores_dir, tables.GRID_IN)
    io.write_table(tables.judge_averages(sets, SC), a.out_dir, "judge_averages")


if __name__ == "__main__":
    main()
