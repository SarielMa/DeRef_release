"""Table 1: derivation gap Gap_tau of the nine judges with Plain input.

Reads the data package (`<data_dir>/<dataset>.jsonl`, `bootstrap_clusters.json`) and the Plain score files
`<scores_dir>/<dataset>/plain/<judge>.jsonl`. Writes `<out_dir>/table1_gap_plain.csv` (proportions) and
`table1_gap_plain_x100.csv`: one row per dataset x judge x scope (out_of_view and each out-of-view path).
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
    io.write_table(tables.table1(sets, a.scores_dir), a.out_dir, "table1_gap_plain")


if __name__ == "__main__":
    main()
