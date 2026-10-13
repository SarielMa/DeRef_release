"""Send judge inputs to Jev (pinned typesafe/jev-1.13-20260917) and record the returned probability.

Environment: JEV_KEY (required), JEV_BASE_URL (default https://openrouter.ai/api); requests go to
{JEV_BASE_URL}/v1/systemone with body {"state": context, "model": <pinned>, "questions": {"q": {"type": "noul",
"instructions": question}}}. The score is answers["q"]["noul"].
Reads a list TSV, one line per batch: <items.jsonl><TAB><output dir> (e.g. <work>/r0.tsv from 44_jev_dedup.py, or a
judge-input file directly). Writes <output dir>/jev.jsonl rows {"item_id", "score", "model", "cached", "latency_ms"}
and <output dir>/jev_failed.jsonl rows {"item_id", "error"}; answers are cached in --cache (a rerun resumes).

Length check (--probe-max): before any request, every input's Qwen3-8B token count of context + "\n" + question
must be <= --probe-max, else the run stops. Default 9000 (all truncated inputs). The full-history inputs were run
with --probe-max 40000 (above Jev's own context), so that Jev's own limit decides; refused inputs
("max_tokens_exceeded") then go down the deletion ladder with 46_jev_ladder_next.py.
Runs as done: Plain-style lists with --half-cost 7.0 --repeat 500 (first run only); later runs with --half-cost 0.

  python scripts/45_score_jev.py --list <work>/r0.tsv --cache <jev_cache.sqlite> --max-calls 200000 --max-cost 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.score import jev_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, help="TSV: <items.jsonl><TAB><output dir> per line")
    ap.add_argument("--cache", required=True, help="sqlite answer cache")
    ap.add_argument("--max-calls", type=int, required=True, help="cap on billed requests in this run")
    ap.add_argument("--max-cost", type=float, required=True, help="cap on cumulative reported cost (dollars)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=0, help="afterwards re-ask N random inputs without the cache")
    ap.add_argument("--state", default=None, help="cumulative billed / cost file (default: next to the cache)")
    ap.add_argument("--stop", default=None, help="stop-flag file (default: next to the cache)")
    ap.add_argument("--half-cost", type=float, default=jev_run.HALF_COST,
                    help="stop if the cost at half of --total-unique billed requests exceeds this; 0 = off")
    ap.add_argument("--total-unique", type=int, default=jev_run.TOTAL_UNIQUE)
    ap.add_argument("--probe-max", type=int, default=jev_run.PROBE_MAX,
                    help="local length check in Qwen3-8B tokens (default %(default)s; full-history inputs: 40000)")
    ap.add_argument("--tokenizer-path", default=None,
                    help="local Qwen3-8B tokenizer directory instead of the pinned HF id + revision")
    jev_run.run(ap.parse_args())


if __name__ == "__main__":
    main()
