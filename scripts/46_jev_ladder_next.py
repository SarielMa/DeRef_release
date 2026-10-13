"""Jev full-history ladder, round R: inputs of round R-1 that Jev refused with "max_tokens_exceeded" get one more
earliest step's return replaced by "  -> [return dropped]" (same renderer as the local judges' full-history input;
a DeRef + full-history input keeps its reference block) and are written for resubmission.

Reads <work>/batch.json, r{R-1}.jsonl, r{R-1}_map.jsonl, out_r{R-1}/, k.json, the data package (history pieces via
deref.inputs.full_text) and the answer cache (read-only). Writes <work>/r{R}.jsonl, r{R}_map.jsonl, r{R}.tsv,
r{R}_counts.json, k.json. Exit 10: nothing left to send; exit 5: refused with all returns dropped (see exhausted.json).
Loop (see scripts/45_jev_full_text.sh): R=1,2,...: 46_jev_ladder_next.py --round R, then 45_score_jev.py --list
<work>/rR.tsv --probe-max 40000, until exit 10.

  python scripts/46_jev_ladder_next.py --work <work> --cache <jev_cache.sqlite> --data-dir <deref_data> --round 1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.score import jev_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--data-dir", required=True, help="data package directory (for the history pieces)")
    ap.add_argument("--round", type=int, required=True, help="round to build (>= 1)")
    a = ap.parse_args()
    assert a.round >= 1
    jev_batch.cmd_next(a.work, a.cache, a.round, a.data_dir)


if __name__ == "__main__":
    main()
