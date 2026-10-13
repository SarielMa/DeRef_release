#!/bin/bash
# Full-history inputs (Plain-full, DeRef + full history) with Qwen3-8B and Qwen3.8-27B on one GPU.
# Usage (from the code package root):
#   JUDGES="qwen3_8b qwen3_8_27b" bash scripts/43_score_local_judge_long.sh <list.tsv>
#   list.tsv: one line per items file: <items.jsonl><TAB><output dir>; writes <output dir>/<judge>.jsonl
# Token budget per batch as run: 65536 for qwen3_8b, 32768 for qwen3_8_27b (the script defaults).
set -euo pipefail
LIST=${1:?usage: 43_score_local_judge_long.sh <list.tsv>}
JUDGES=${JUDGES:-"qwen3_8b qwen3_8_27b"}
for J in ${JUDGES}; do
  while IFS=$'\t' read -r IT OUT; do
    echo "=== ${J}: ${IT}"
    python scripts/43_score_local_judge_long.py --judge "${J}" --items "${IT}" --out "${OUT}/${J}.jsonl"
  done < "${LIST}"
done
echo "=== done"
