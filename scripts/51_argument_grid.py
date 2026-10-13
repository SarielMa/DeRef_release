"""Argument-level grid and paired differences of the four Table 2 judges (supports Table 2; per-judge AUROC).

Reads the data package and the score files of all grid inputs (`<scores_dir>/<dataset>/<input>/<judge>.jsonl`).
Writes `argument_grid.csv` (dataset x judge x input x path: n, tau, q, AUROC, accepted shares, Gap_tau, with CIs) and
`argument_diffs.csv` (paired differences of AUROC and Gap_tau with CIs and the SD of their draws), each also as `_x100`.
Runtime about 11 minutes on 4 CPU cores.
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
    SC, NOTE = load.load_scores(a.scores_dir, tables.GRID_IN)
    R, D = tables.grid(sets, SC, NOTE, log=io.log)
    io.write_table(R, a.out_dir, "argument_grid")
    io.write_table(D, a.out_dir, "argument_diffs")


if __name__ == "__main__":
    main()
