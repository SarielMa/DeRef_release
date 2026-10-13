#!/bin/bash
# Local Yes/No judges on one GPU (bf16; Qwen3-32B and Qwen3.8-27B need a large-memory GPU, the paper used one B200).
# Usage (from the code package root):
#   JUDGES="qwen3_8b qwen3_8_27b" bash scripts/42_score_local_judges.sh <list.tsv>
#   list.tsv: one line per items file: <items.jsonl><TAB><output dir>; writes <output dir>/<judge>.jsonl
# Batch size: the default of 42_score_local_judge.py (32) is what the paper runs used.
set -euo pipefail
LIST=${1:?usage: 42_score_local_judges.sh <list.tsv>}
JUDGES=${JUDGES:?set JUDGES, e.g. "qwen3_8b qwen3_8_27b"}
for J in ${JUDGES}; do
  while IFS=$'\t' read -r IT OUT; do
    python scripts/40_check_lengths.py --judge "${J}" --items "${IT}"
    echo "=== ${J}: ${IT}"
    python scripts/42_score_local_judge.py --judge "${J}" --items "${IT}" --out "${OUT}/${J}.jsonl"
  done < "${LIST}"
done
echo "=== done"
