"""FCRB-Relocate step 1: choose one relocatable argument per FC-RewardBench state and write the request-rewrite prompts.

Reads <data-dir>/data/fcrb.jsonl. Writes <out> (rows {"prompt_id" = state id, "idx", "text", "tokens", "d", "gold_attrs",
candidate fields}) and <out stem>_candidates_by_param.json. Needs the Qwen3.8-27B tokenizer (chat template).

  python scripts/60_relocate_rewrite_prompts.py --data-dir <deref_data> --out <relocate_dir>/rewrite_prompts.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.build import relocate as RL  # noqa: E402
from deref.common import write_jsonl  # noqa: E402
from deref.inputs import state as ST  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--out", required=True, help="output prompt file (.jsonl)")
    ap.add_argument("--tokenizer-qwen3-8-27b", default=None, help="local tokenizer directory (default: Hugging Face id)")
    a = ap.parse_args()
    tok = ST.load_tokenizer(ST.QWEN3_8_27B, a.tokenizer_qwen3_8_27b)
    rows, byp = RL.rewrite_prompts(ST.load_records(a.data_dir, "fcrb"), tok)
    out = Path(a.out)
    write_jsonl(out, rows)
    out.with_name(out.stem + "_candidates_by_param.json").write_text(json.dumps(byp, ensure_ascii=False))
    print(f"[relocate] {sum(n for _, n in byp)} candidate arguments; {len(rows)} prompts (one per state)")


if __name__ == "__main__":
    main()
