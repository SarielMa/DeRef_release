"""Build the reference-swap judge inputs of Figure 3.

Reads the data package and the DeRef judge items <inputs-dir>/deref_<dataset>.jsonl (from 13_build_reference_items.py
--method deref) and the Qwen3-8B tokenizer (CLM length). Writes <out>/swap_units.jsonl (one row per unit: gold, the
swapped-in value and where it comes from, the reference before / after), <out>/deref_swap_<dataset>.jsonl (the swapped
judge items to score, ids "<argument item_id>|deref_swap|<unit>"; store their scores as <scores_dir>/<dataset>/deref_swap/<judge>.jsonl), <out>/deref_swap_original.jsonl
(the matching unswapped DeRef items, for reference; their scores are the DeRef scores) and <out>/swap_stats.json.
CPU, a few minutes.

  python scripts/72_build_swap.py --data-dir <deref_data> --inputs-dir <inputs_dir> --out <inputs_dir>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import read_jsonl, write_jsonl  # noqa: E402
from deref.inputs import state as ST  # noqa: E402
from deref.inputs import swap as SW  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--inputs-dir", required=True, help="directory with deref_<dataset>.jsonl")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--tokenizer-qwen3-8b", default=None, help="local tokenizer directory (default: Hugging Face id)")
    a = ap.parse_args()
    recs = {d: ST.load_records(a.data_dir, d) for d in SW.DATASETS}
    deref = {d: {r["base_item"]: r for r in read_jsonl(Path(a.inputs_dir) / f"deref_{d}.jsonl")} for d in SW.DATASETS}
    tok = ST.load_tokenizer(ST.QWEN3_8B, a.tokenizer_qwen3_8b)
    units, swap_rows, orig_rows, st = SW.build(recs, deref, tok)
    out = Path(a.out)
    write_jsonl(out / "swap_units.jsonl", units)
    for d in SW.DATASETS:
        write_jsonl(out / f"{SW.TAG}_{d}.jsonl", [r for r in swap_rows if r["dataset"] == d])
    write_jsonl(out / f"{SW.TAG}_original.jsonl", orig_rows)
    (out / "swap_stats.json").write_text(json.dumps(st, indent=1) + "\n")
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
