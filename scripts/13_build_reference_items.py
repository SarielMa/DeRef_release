"""Build the reference-reading judge inputs of one method from its generations.

Methods: deref, revise, evidence_search, sgv_prior, deref_nothink, deref_qwen3_8b, expected_value (expected_value
reads the deref generations and the expected-value parser, deref/inputs/expected_value_parser.py).
Reads <data-dir>/data/<dataset>.jsonl, <gen-dir>/generations/<generation method>.jsonl (rows {"prompt_id", "output"}),
<gen-dir>/prompts/<generation method>_map.jsonl (rows {"item_id": argument item_id, "prompt_id"}) and the Qwen3-8B
tokenizer (CLM length, 2,048-token ladder).
Writes <out>/<method>_<dataset>.jsonl, rows {"item_id": "<argument item_id>|<tag>", "base_item", "context", "question",
"clm_tokens", "truncation", "extra_truncation", ...}, and <out>/<method>_<dataset>_plain_kept.jsonl (arguments of the
map without an own item: no answer, no snippet, or over 2,048 tokens at every rung; they keep the Plain score).

  python scripts/13_build_reference_items.py --method deref --data-dir <deref_data> --gen-dir <gen_dir> --out <inputs_dir>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import write_jsonl  # noqa: E402
from deref.inputs import reference as RF  # noqa: E402
from deref.inputs import state as ST  # noqa: E402

GENERATION_OF = {"expected_value": "deref"}  # methods without their own generation


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=sorted(RF.METHODS))
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--gen-dir", required=True, help="directory with generations/ and prompts/")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--datasets", nargs="+", default=list(ST.DATASETS), choices=ST.DATASETS)
    ap.add_argument("--all-arguments", action="store_true", help="also arguments outside the analysis set")
    ap.add_argument("--tokenizer-qwen3-8b", default=None, help="local tokenizer directory (default: Hugging Face id)")
    a = ap.parse_args()
    g = GENERATION_OF.get(a.method, a.method)
    gens = RF.load_generations(Path(a.gen_dir) / "generations" / f"{g}.jsonl")
    mp = RF.load_map(Path(a.gen_dir) / "prompts" / f"{g}_map.jsonl")
    tok = ST.load_tokenizer(ST.QWEN3_8B, a.tokenizer_qwen3_8b)
    for d in a.datasets:
        rows, kept = RF.build(a.method, ST.load_records(a.data_dir, d), gens, mp, tok, analysis_only=not a.all_arguments)
        write_jsonl(Path(a.out) / f"{a.method}_{d}.jsonl", rows)
        write_jsonl(Path(a.out) / f"{a.method}_{d}_plain_kept.jsonl", [{"item_id": k, "reason": r} for k, r in kept])
        print(f"[{a.method}] {d}: {len(rows)} items, {len(kept)} keep Plain")


if __name__ == "__main__":
    main()
