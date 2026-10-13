"""Jev round 0: one request row per distinct judge input over all batches, minus inputs already in the answer cache.

Reads a batch list TSV, one line per judge-input file:
  <items.jsonl><TAB><output score file>[<TAB><dataset><TAB><kind>]
  (kind = plain | plain_full | deref_full; dataset and kind are needed only for the full-history ladder)
and the sqlite answer cache (read-only). Writes into --work: batch.json, r0.jsonl (inputs to send), r0_map.jsonl
(item -> input), r0.tsv (list for 45_score_jev.py), r0_counts.json.
Then: 45_score_jev.py --list <work>/r0.tsv; for full-history inputs 46_jev_ladder_next.py; finally 47_jev_final.py.

  python scripts/44_jev_dedup.py --batches jev_batches.tsv --work <work> --cache <jev_cache.sqlite>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.score import jev_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", required=True, help="batch list TSV")
    ap.add_argument("--work", required=True, help="work directory for the Jev rounds")
    ap.add_argument("--cache", required=True, help="sqlite answer cache (created by 45_score_jev.py if absent)")
    a = ap.parse_args()
    if not Path(a.cache).exists():
        import sqlite3
        db = sqlite3.connect(a.cache)
        db.execute("CREATE TABLE IF NOT EXISTS ans (k TEXT PRIMARY KEY, v TEXT)")
        db.close()
    jev_batch.cmd_dedup(a.batches, a.work, a.cache)


if __name__ == "__main__":
    main()
