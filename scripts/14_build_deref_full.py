"""Build the DeRef + full-history judge inputs: the Plain-full input with the DeRef reference block (HEAD + answer)
before the argument tail. ComplexFuncBench: block pasted without re-running the ladder (error if over 32,768 tokens);
FinTrace: the 32,768-token ladder is re-run with the block. FC-RewardBench has no history (its input is the deref one).

Reads <data-dir>/data/{cfb,fintrace}.jsonl, <gen-dir>/generations/deref.jsonl, <gen-dir>/prompts/deref_map.jsonl and the
Qwen3-8B / Qwen3.8-27B tokenizers. Writes <out>/deref_full_<dataset>.jsonl, rows {"item_id" (argument item_id),
"context", "question", "tokens", "cut", "level", "level_plain_full", "state"}; arguments without an answer keep their
Plain-full input.

  python scripts/14_build_deref_full.py --data-dir <deref_data> --gen-dir <gen_dir> --out <inputs_dir>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import write_jsonl  # noqa: E402
from deref.inputs import full_text as FT  # noqa: E402
from deref.inputs import reference as RF  # noqa: E402
from deref.inputs import state as ST  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--gen-dir", required=True, help="directory with generations/deref.jsonl and prompts/deref_map.jsonl")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--datasets", nargs="+", default=["cfb", "fintrace"], choices=ST.DATASETS)
    ap.add_argument("--all-arguments", action="store_true", help="also arguments outside the analysis set")
    ap.add_argument("--tokenizer-qwen3-8b", default=None, help="local tokenizer directory (default: Hugging Face id)")
    ap.add_argument("--tokenizer-qwen3-8-27b", default=None, help="local tokenizer directory (default: Hugging Face id)")
    a = ap.parse_args()
    gens = RF.load_generations(Path(a.gen_dir) / "generations" / "deref.jsonl")
    mp = RF.load_map(Path(a.gen_dir) / "prompts" / "deref_map.jsonl")
    b = FT.Builder(ST.load_tokenizer(ST.QWEN3_8B, a.tokenizer_qwen3_8b),
                   ST.load_tokenizer(ST.QWEN3_8_27B, a.tokenizer_qwen3_8_27b))
    for d in a.datasets:
        rows, kept = [], []
        for rec, arg in ST.arguments(ST.load_records(a.data_dir, d), not a.all_arguments):
            if rec["dataset"] == "fcrb" or arg["item_id"] not in mp:
                continue
            out = gens.get(mp[arg["item_id"]])
            block = RF.deref_full_block(out) if out is not None else None
            if block is None:
                kept.append({"item_id": arg["item_id"], "reason": "no answer"})
                continue
            rows.append(FT.deref_full_item(b, rec, arg, block))
        write_jsonl(Path(a.out) / f"deref_full_{d}.jsonl", rows)
        write_jsonl(Path(a.out) / f"deref_full_{d}_plain_full_kept.jsonl", kept)
        print(f"[deref_full] {d}: {len(rows)} items, {len(kept)} keep Plain-full")


if __name__ == "__main__":
    main()
