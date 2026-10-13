"""FCRB-Relocate step 3: build the states (both versions) from the rewrites and write the data-package file.

Reads <data-dir>/data/fcrb.jsonl, the state export of step 07 (<work-dir>/states/export.jsonl; distractor value pool),
the prompts of step 1 and the generations of step 2. Optionally checks that the added function name does not occur in
the FC-RewardBench source files (--fcrb-raw). Writes <out> (records `fcrb_relocate`, one per state, version and
candidate role) and, with --debug-dir, states.jsonl / items.jsonl / drops.json (the Plain items of every argument).
Needs the Qwen3-8B tokenizer (2,048-token check).

  python scripts/62_relocate_build.py --data-dir <deref_data> --export <W>/states/export.jsonl \\
      --prompts <relocate_dir>/rewrite_prompts.jsonl --generations <relocate_dir>/rewrite_generations.jsonl \\
      --out <deref_data>/data/fcrb_relocate.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.build import relocate as RL  # noqa: E402
from deref.common import read_jsonl, write_jsonl  # noqa: E402
from deref.inputs import state as ST  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--export", required=True, help="state export of step 07 (export.jsonl)")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--out", required=True, help="output record file (fcrb_relocate.jsonl)")
    ap.add_argument("--fcrb-raw", default=None, help="FC-RewardBench source directory (optional name-collision check)")
    ap.add_argument("--debug-dir", default=None, help="also write states / items / drop counts here")
    ap.add_argument("--tokenizer-qwen3-8b", default=None, help="local tokenizer directory (default: Hugging Face id)")
    a = ap.parse_args()
    records = ST.load_records(a.data_dir, "fcrb")
    raw = None
    if a.fcrb_raw:
        raw = b"".join(f.read_bytes() for f in sorted(Path(a.fcrb_raw).rglob("*")) if f.is_file())
    gens = {r["prompt_id"]: r for r in read_jsonl(a.generations)}
    states, items, why = RL.build_states(records, read_jsonl(a.export), read_jsonl(a.prompts), gens,
                                         ST.load_tokenizer(ST.QWEN3_8B, a.tokenizer_qwen3_8b), raw)
    recs = RL.to_records(states, records)
    write_jsonl(a.out, recs)
    if a.debug_dir:
        d = Path(a.debug_dir)
        write_jsonl(d / "states.jsonl", states)
        write_jsonl(d / "items.jsonl", items)
        (d / "drops.json").write_text(json.dumps(dict(why), indent=1))
    print(f"[relocate] {len(states)} states, {len(recs)} records; dropped {sum(why.values())}: {dict(why)}")


if __name__ == "__main__":
    main()
