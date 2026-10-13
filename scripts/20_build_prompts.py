"""Build the generation prompts of every method from the data package (CPU; Qwen3.8-27B and Qwen3-8B tokenizers).

Reads  <data_dir>/data/{fcrb,cfb,fintrace}.jsonl (data-package records; <data_dir> may also be the data/ folder).
Writes <out_dir>/<method>.jsonl      rows {"prompt_id", "text", "tokens", "dataset"} (all datasets in one file)
       <out_dir>/<method>_map.jsonl  rows {"item_id", "prompt_id"} for every in-analysis argument the method covers.

Methods (see deref/generate/prompts.py): the paper's thinking-on prompts deref (DeRef), revise (RevisEval),
evidence_search (ProvenanceGuard evidence), sgv_prior (Table 3 "w/o full history"), deref_qwen3_8b (Table 3 "Qwen3-8B
as deriver"), and their thinking-off originals <method>_nothink
(deref_nothink is the "thinking off" derivation of Table 3). Generate the thinking-on files with
`22_generate.py --mode thinking` (deref_qwen3_8b with `--model qwen3_8b`), the *_nothink files with `--mode greedy`.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import write_jsonl  # noqa: E402
from deref.generate import prompts as GP  # noqa: E402
from deref.generate.models import load_tokenizer  # noqa: E402
from deref.inputs.state import load_records  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="data package directory (its data/ folder or the folder itself)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", default="fcrb,cfb,fintrace")
    ap.add_argument("--methods", default=",".join(GP.METHODS), help="comma-separated method names")
    a = ap.parse_args()
    methods = tuple(a.methods.split(","))
    unknown = set(methods) - set(GP.METHODS + (GP.LOCATOR,))
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}")
    tok27 = load_tokenizer("qwen3_8_27b")
    tok8 = load_tokenizer("qwen3_8b") if {"deref_qwen3_8b", "deref_qwen3_8b_nothink"} & set(methods) else None
    rows = {m: ([], []) for m in methods}
    for d in a.datasets.split(","):
        recs = load_records(a.data_dir, d)
        for m, (p, mp) in GP.build(recs, d, tok27, tok8, methods=methods).items():
            rows[m][0].extend(p)
            rows[m][1].extend(mp)
            print(f"[{m} {d}] {len(p)} prompts, {len(mp)} arguments, max tokens {max((r['tokens'] for r in p), default=0)}",
                  flush=True)
    out = Path(a.out_dir)
    for m, (p, mp) in rows.items():
        write_jsonl(out / f"{m}.jsonl", p)
        write_jsonl(out / f"{m}_map.jsonl", mp)


if __name__ == "__main__":
    main()
