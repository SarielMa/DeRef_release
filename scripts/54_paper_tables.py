"""Paper-style views of Tables 1-3 (x100, two decimals, "estimate [lo, hi]").

Reads from --in-dir: `table1_gap_plain.csv` (step 50), `argument_grid.csv` and `argument_diffs.csv` (step 51),
`ablation.csv` (step 53) and, if present, `table2_self_reference.csv` (step 86). Writes `paper_table1.csv` (Gap_tau out of
view and on each dataset's main path), `paper_table2.csv` (argument AUROC per judge and input, and DeRef - RevisEval)
and `paper_table3.csv` (ablation on --table3-dataset, default cfb) to --out-dir.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.stats import io, tables  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="directory with the CSV files of steps 50, 51, 53 (and 86)")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--table3-dataset", default="cfb", choices=("fcrb", "cfb", "fintrace"))
    a = ap.parse_args()
    i, o = Path(a.in_dir), Path(a.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    tables.paper_table1(pd.read_csv(i / "table1_gap_plain.csv")).to_csv(o / "paper_table1.csv", index=False)
    sr = i / "table2_self_reference.csv"
    tables.paper_table2(pd.read_csv(i / "argument_grid.csv"), pd.read_csv(i / "argument_diffs.csv"),
                        pd.read_csv(sr) if sr.exists() else None).to_csv(o / "paper_table2.csv", index=False)
    tables.paper_table3(pd.read_csv(i / "ablation.csv"), a.table3_dataset).to_csv(o / "paper_table3.csv", index=False)
    io.log(f"wrote paper_table1.csv, paper_table2.csv, paper_table3.csv to {o}")


if __name__ == "__main__":
    main()
