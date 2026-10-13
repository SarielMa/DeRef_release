"""Build the schema-facts judge inputs of Figure 2b (FC-RewardBench arguments whose schema has a default).

Reads <data-dir>/data/fcrb.jsonl. Writes <out>/schema_facts_fcrb.jsonl, rows {"item_id": "<argument item_id>|schema_facts",
"base_item", "context", "question", "label", "path", "hit", "value", "default"}. Score them like any other input and
store the scores as <scores_dir>/fcrb/schema_facts/<judge>.jsonl (arguments without a schema-facts row keep their Plain
score). CPU, seconds.

  python scripts/70_build_schema_facts.py --data-dir <deref_data> --out <inputs_dir>
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import write_jsonl  # noqa: E402
from deref.inputs import schema_facts as SF  # noqa: E402
from deref.inputs import state as ST  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--all-arguments", action="store_true", help="also arguments outside the analysis set")
    a = ap.parse_args()
    rows = SF.build(ST.load_records(a.data_dir, "fcrb"), analysis_only=not a.all_arguments)
    write_jsonl(Path(a.out) / f"{SF.TAG}_fcrb.jsonl", rows)
    c = Counter((r["path"], r["label"], r["hit"]) for r in rows)
    print(f"[schema_facts] fcrb: {len(rows)} items")
    for k, v in sorted(c.items(), key=str):
        print(f"  path={k[0]} label={k[1]} hit={k[2]}: {v}")


if __name__ == "__main__":
    main()
