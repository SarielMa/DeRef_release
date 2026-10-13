"""Figure 2b: acceptance at tau of correct FC-RewardBench defaults (Plain / Schema facts / DeRef, four judges).

Reads the data package, the DeRef judge items (<inputs-dir>/deref_fcrb.jsonl, from 13_build_reference_items.py), the
schema-facts items (<inputs-dir>/schema_facts_fcrb.jsonl, from 70_build_schema_facts.py) and the score files
<scores_dir>/fcrb/{plain,schema_facts,deref}/<judge>.jsonl. Writes figure2b_default.csv (proportions) and
figure2b_default_x100.csv. The figure shows the `reference_says_default` rows (n = 142). CPU, seconds.

  python scripts/71_figure2b.py --data-dir <deref_data> --inputs-dir <inputs_dir> --scores-dir <scores_dir> --out-dir <tables_dir>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import read_jsonl  # noqa: E402
from deref.stats import figure2b as F2  # noqa: E402
from deref.stats import load  # noqa: E402
from deref.stats.tables import times100  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--inputs-dir", required=True, help="directory with deref_fcrb.jsonl and schema_facts_fcrb.jsonl")
    ap.add_argument("--scores-dir", required=True, help="score files <scores_dir>/<dataset>/<input>/<judge>.jsonl")
    ap.add_argument("--out-dir", required=True, help="output directory")
    a = ap.parse_args()
    fc = load.load_sets(a.data_dir)["fcrb"]
    deref = read_jsonl(Path(a.inputs_dir) / "deref_fcrb.jsonl")
    sf = read_jsonl(Path(a.inputs_dir) / "schema_facts_fcrb.jsonl")
    df = F2.table(fc, a.scores_dir, deref, sf)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "figure2b_default.csv", index=False)
    times100(df, ["acc", "se", "lo", "hi"]).to_csv(out / "figure2b_default_x100.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
