#!/bin/bash
# Jev on the full-history inputs (CPU only; needs JEV_KEY, optional JEV_BASE_URL).
# Round 0 sends every distinct input not yet in the cache with the local length check raised to 40,000 Qwen3-8B
# tokens (--probe-max 40000), so Jev's own context limit decides. Inputs Jev refuses ("max_tokens_exceeded") go one
# more step down the deletion ladder and are resubmitted until none is refused (exit 10) or an item is refused with
# all returns dropped (exit 5: stop and inspect <work>/exhausted.json). Then the final per-item score files.
# Usage (from the code package root):
#   bash scripts/45_jev_full_text.sh <batches.tsv> <work dir> <jev_cache.sqlite> <data dir>
#   batches.tsv: <items.jsonl><TAB><output score file><TAB><dataset><TAB><plain_full|deref_full> per line
set -uo pipefail
BATCHES=${1:?batches.tsv}; WORK=${2:?work dir}; CACHE=${3:?cache}; DATA=${4:?data dir}
: "${JEV_KEY:?set JEV_KEY}"
JEV="python scripts/45_score_jev.py --cache ${CACHE} --max-cost 10.0 --workers 8 --state ${WORK}/run_state.json --stop ${WORK}/STOP --half-cost 0 --probe-max 40000"
python scripts/44_jev_dedup.py --batches "${BATCHES}" --work "${WORK}" --cache "${CACHE}" || exit 1
${JEV} --list "${WORK}/r0.tsv" --max-calls 27000 --total-unique 26683
echo "round 0 exit=$?"
R=1
while [ $R -le 30 ]; do
  python scripts/46_jev_ladder_next.py --work "${WORK}" --cache "${CACHE}" --data-dir "${DATA}" --round $R
  rc=$?
  if [ $rc -eq 10 ]; then echo "=== ladder done after round $((R-1))"; break; fi
  if [ $rc -ne 0 ]; then echo "=== ladder STOPPED at round $R (exit $rc)"; exit $rc; fi
  ${JEV} --list "${WORK}/r${R}.tsv" --max-calls 5000 --total-unique 5000
  echo "round $R exit=$?"
  R=$((R+1))
done
python scripts/47_jev_final.py --work "${WORK}" --cache "${CACHE}" --data-dir "${DATA}"
