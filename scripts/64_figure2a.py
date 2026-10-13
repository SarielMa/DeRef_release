"""Figure 2a: argument AUROC on FCRB-Relocate (visible / buried version x judge x Plain, full history, DeRef).

Reads <data-dir>/data/fcrb_relocate.jsonl and <scores-dir>/fcrb_relocate/{plain,plain_full,deref}/<judge>.jsonl
(rows {"item_id", "score"}). Writes <out-dir>/figure2a_relocate.csv (proportions) and figure2a_relocate_x100.csv.

  python scripts/64_figure2a.py --data-dir <deref_data> --scores-dir <S> --out-dir <T>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.stats import relocate as SR  # noqa: E402
from deref.stats.tables import times100  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--scores-dir", required=True, help="score files <scores_dir>/fcrb_relocate/<input>/<judge>.jsonl")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    df = SR.figure2a(a.data_dir, a.scores_dir)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "figure2a_relocate.csv", index=False)
    times100(df, ["auroc", "se", "lo", "hi"]).to_csv(out / "figure2a_relocate_x100.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
