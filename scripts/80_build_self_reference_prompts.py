"""Build the generation prompts of the self-generated reference baseline (Table 2) for one judge model.

The judge model writes the expected value from its own truncated (Plain) input: system "You are a helpful assistant that
responds to the user in a concise way.", user = Plain context without the value line + "Give the value of parameter
<name> for this call.", the judge's chat template with thinking off. One prompt per locator slot (shared by its candidates).
Reads <data-dir>/data/<dataset>.jsonl and the judge's tokenizer. Writes <out>/self_reference_<judge>.jsonl (rows
{"prompt_id", "text", "tokens", "dataset"}) and <out>/self_reference_<judge>_map.jsonl (rows {"item_id", "prompt_id"}).

  python scripts/80_build_self_reference_prompts.py --judge qwen3_8b --data-dir <deref_data> --out <gen_dir>/prompts
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import write_jsonl  # noqa: E402
from deref.generate import self_reference as SR  # noqa: E402
from deref.inputs import state as ST  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", required=True, choices=sorted(SR.JUDGE_MODEL))
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--out", required=True, help="output directory (e.g. <gen_dir>/prompts)")
    ap.add_argument("--datasets", nargs="+", default=list(ST.DATASETS), choices=ST.DATASETS)
    ap.add_argument("--tokenizer-path", default=None, help="local tokenizer directory of the judge model")
    a = ap.parse_args()
    from deref.generate.models import MODELS
    from transformers import AutoTokenizer
    m = MODELS[SR.JUDGE_MODEL[a.judge]]
    tok = AutoTokenizer.from_pretrained(a.tokenizer_path) if a.tokenizer_path else \
        AutoTokenizer.from_pretrained(m["repo_id"], revision=m["revision"])
    prompts, maps = [], []
    for d in a.datasets:
        p, mp = SR.build_prompts(ST.load_records(a.data_dir, d), d, tok, a.judge)
        prompts += p
        maps += mp
        print(f"[self-reference] {a.judge} {d}: {len(mp)} arguments, {len(p)} prompts")
    write_jsonl(Path(a.out) / f"{SR.TAG}_{a.judge}.jsonl", prompts)
    write_jsonl(Path(a.out) / f"{SR.TAG}_{a.judge}_map.jsonl", maps)


if __name__ == "__main__":
    main()
