#!/bin/bash
# Qwen3.8-27B judge with thinking on for Table 4 (one large-memory GPU per shard; the paper used one B200 per shard).
# Usage (from the code package root): bash scripts/83_score_local_judge_thinking.sh <inputs_dir> <scores_dir> <work_dir>
#   <inputs_dir> holds plain_cfb.jsonl, plain_full_cfb.jsonl, deref_full_cfb.jsonl, deref_cfb.jsonl (scripts 10, 12-14).
set -euo pipefail
I=${1:?inputs dir}; S=${2:?scores dir}; W=${3:?work dir}
python scripts/83_score_local_judge_thinking.py dedup --work "${W}" \
  --items "${I}/plain_cfb.jsonl" "${I}/plain_full_cfb.jsonl" "${I}/deref_full_cfb.jsonl" "${I}/deref_cfb.jsonl"
python scripts/83_score_local_judge_thinking.py check --work "${W}"
python scripts/83_score_local_judge_thinking.py shard --work "${W}" --n 2
for s in 0 1; do   # the two shards can run as separate jobs
  python scripts/83_score_local_judge_thinking.py gen --work "${W}" --shard ${s}
  python scripts/83_score_local_judge_thinking.py score --work "${W}" --shard ${s}
done
for inp in plain plain_full deref_full deref; do
  python scripts/83_score_local_judge_thinking.py expand --work "${W}" --items "${I}/${inp}_cfb.jsonl" \
    --out "${S}/cfb/${inp}/qwen3_8_27b_thinking.jsonl"
done
