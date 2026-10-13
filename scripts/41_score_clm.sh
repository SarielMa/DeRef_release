#!/bin/bash
# CLM scoring on one GPU: start the vLLM pooling server for Qwen3-8B, run the length checks, score every items file.
# Usage (from the code package root):
#   CLM_CKPT=<CLM_v0.1-8B.pt> bash scripts/41_score_clm.sh <list.tsv>
#   list.tsv: one line per items file: <items.jsonl><TAB><output score file>
# MAX_MODEL_LEN: the Table 1 Plain run used 2048 (the default); all later CLM runs used 2064, which avoids a vLLM
# hang on inputs of exactly 2,048 tokens under concurrent requests. The client truncates at 2,048 in both cases.
set -euo pipefail
LIST=${1:?usage: 41_score_clm.sh <list.tsv>}
: "${CLM_CKPT:?set CLM_CKPT to the CLM head checkpoint}"
export CLM_EMB_URL=${CLM_EMB_URL:-http://localhost:8090/v1/embeddings}
QWEN_REV=b968826d9c46dd6066d109eabc6255188de91218
LOGDIR=${LOGDIR:-logs}
mkdir -p "${LOGDIR}"

while IFS=$'\t' read -r IT OUT; do python scripts/40_check_lengths.py --judge clm --items "${IT}"; done < "${LIST}"

vllm serve Qwen/Qwen3-8B --revision "${QWEN_REV}" --served-model-name qwen3-8b --runner pooling \
  --pooler-config '{"seq_pooling_type": "LAST"}' --max-model-len "${MAX_MODEL_LEN:-2048}" --dtype bfloat16 \
  --gpu-memory-utilization 0.5 --port 8090 > "${LOGDIR}/vllm_server.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  if curl -sf http://localhost:8090/health >/dev/null; then echo "server up after $((i * 5))s"; break; fi
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then echo "vLLM server died"; tail -50 "${LOGDIR}/vllm_server.log"; exit 1; fi
  sleep 5
done

while IFS=$'\t' read -r IT OUT; do
  python scripts/40_check_lengths.py --judge clm --items "${IT}" --server "${CLM_EMB_URL}"
  echo "=== clm: ${IT}"
  python scripts/41_score_clm.py --items "${IT}" --out "${OUT}" --checkpoint "${CLM_CKPT}" --emb-url "${CLM_EMB_URL}"
done < "${LIST}"
echo "=== done"
