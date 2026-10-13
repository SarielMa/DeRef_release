"""FCRB-Relocate judge inputs and DeRef prompts (Figure 2a).

  plain       -> <out>/plain_fcrb_relocate.jsonl            (rows {"item_id", "context", "question"})
  plain_full  -> <out>/plain_full_fcrb_relocate.jsonl       (Qwen3-8B and Qwen3.8-27B tokenizers, 32,768-token ladder)
  prompts     -> <gen-dir>/prompts/deref_fcrb_relocate.jsonl and deref_fcrb_relocate_map.jsonl (Qwen3.8-27B tokenizer);
                 generate them with scripts/22_generate.py --mode thinking --max-new 8192
  deref       -> <out>/deref_fcrb_relocate.jsonl (+ _plain_kept.jsonl); reads
                 <gen-dir>/generations/deref_fcrb_relocate.jsonl and the map (Qwen3-8B tokenizer)

  python scripts/63_relocate_inputs.py plain --data-dir <deref_data> --out <inputs_dir>
  python scripts/63_relocate_inputs.py prompts --data-dir <deref_data> --gen-dir <gen_dir>
  python scripts/63_relocate_inputs.py deref --data-dir <deref_data> --gen-dir <gen_dir> --out <inputs_dir>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import write_jsonl  # noqa: E402
from deref.inputs import full_text as FT  # noqa: E402
from deref.inputs import reference as RF  # noqa: E402
from deref.inputs import relocate as RI  # noqa: E402
from deref.inputs import state as ST  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("plain", "plain_full", "prompts", "deref"))
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--out", default=None, help="output directory of the judge inputs")
    ap.add_argument("--gen-dir", default=None, help="directory with prompts/ and generations/")
    ap.add_argument("--tokenizer-qwen3-8b", default=None, help="local tokenizer directory (default: Hugging Face id)")
    ap.add_argument("--tokenizer-qwen3-8-27b", default=None, help="local tokenizer directory (default: Hugging Face id)")
    a = ap.parse_args()
    recs = RI.load_records(a.data_dir)
    name = RI.DATASET
    if a.step == "plain":
        rows = RI.plain_rows(recs)
        write_jsonl(Path(a.out) / f"plain_{name}.jsonl", rows)
    elif a.step == "plain_full":
        b = FT.Builder(ST.load_tokenizer(ST.QWEN3_8B, a.tokenizer_qwen3_8b),
                       ST.load_tokenizer(ST.QWEN3_8_27B, a.tokenizer_qwen3_8_27b))
        rows = RI.plain_full_rows(recs, b)
        write_jsonl(Path(a.out) / f"plain_full_{name}.jsonl", rows)
    elif a.step == "prompts":
        rows, mp = RI.deref_prompts(recs, ST.load_tokenizer(ST.QWEN3_8_27B, a.tokenizer_qwen3_8_27b))
        write_jsonl(Path(a.gen_dir) / "prompts" / f"deref_{name}.jsonl", rows)
        write_jsonl(Path(a.gen_dir) / "prompts" / f"deref_{name}_map.jsonl", mp)
    else:
        g = Path(a.gen_dir)
        rows, kept = RI.deref_rows(recs, RF.load_generations(g / "generations" / f"deref_{name}.jsonl"),
                                   RF.load_map(g / "prompts" / f"deref_{name}_map.jsonl"),
                                   ST.load_tokenizer(ST.QWEN3_8B, a.tokenizer_qwen3_8b))
        write_jsonl(Path(a.out) / f"deref_{name}.jsonl", rows)
        write_jsonl(Path(a.out) / f"deref_{name}_plain_kept.jsonl", [{"item_id": k, "reason": r} for k, r in kept])
    print(f"[relocate {a.step}] {len(rows)} rows")


if __name__ == "__main__":
    main()
