"""Figure 3: acceptance at 0.5 and argument AUROC with the original vs the swapped DeRef reference (four judges).

Reads the data package, the swapped items <inputs-dir>/deref_swap_<dataset>.jsonl (from 72_build_swap.py) and the score
files <scores_dir>/<dataset>/{plain,deref,deref_swap}/<judge>.jsonl. Writes figure3_swap.csv (proportions) and
figure3_swap_x100.csv. CPU, about a minute.

  python scripts/73_figure3.py --data-dir <deref_data> --inputs-dir <inputs_dir> --scores-dir <scores_dir> --out-dir <tables_dir>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import read_jsonl  # noqa: E402
from deref.stats import figure3 as F3  # noqa: E402
from deref.stats import load  # noqa: E402
from deref.stats.tables import times100  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--inputs-dir", required=True, help="directory with deref_swap_<dataset>.jsonl")
    ap.add_argument("--scores-dir", required=True, help="score files <scores_dir>/<dataset>/<input>/<judge>.jsonl")
    ap.add_argument("--out-dir", required=True, help="output directory")
    a = ap.parse_args()
    sets = load.load_sets(a.data_dir)
    items = {d: read_jsonl(Path(a.inputs_dir) / f"{F3.TAG}_{d}.jsonl") for d in F3.DATASETS}
    df = F3.table(sets, a.scores_dir, items)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "figure3_swap.csv", index=False)
    times100(df, ["orig", "orig_se", "swap", "swap_se", "delta", "lo", "hi", "se_delta"]).to_csv(out / "figure3_swap_x100.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
