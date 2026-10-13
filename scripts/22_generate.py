"""Generate one output per prompt with vLLM (GPU; see deref/generate/vllm_gen.py for the exact decoding).

Reads  a prompt file of 20_build_prompts.py.
Writes rows {"prompt_id", "output", "new_tokens", "finished", "finish_reason", "prompt_tokens"} (+ "think_end_index"
       with --mode thinking).

As run for the paper:
  thinking-on methods:  --mode thinking --max-new 8192                      (deref_qwen3_8b: add --model qwen3_8b)
  thinking-off originals (*_nothink): --mode greedy --max-new 1024           (evidence_search_nothink: --max-new 160;
                                                                              deref_qwen3_8b_nothink: --model qwen3_8b)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.generate.vllm_gen import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3_8_27b", choices=("qwen3_8_27b", "qwen3_8b"))
    ap.add_argument("--mode", default="thinking", choices=("thinking", "greedy"))
    ap.add_argument("--max-new", type=int, default=None, help="new-token limit (default: 8192 thinking, 160 greedy)")
    a = ap.parse_args()
    max_new = a.max_new if a.max_new is not None else (8192 if a.mode == "thinking" else 160)
    generate(a.prompts, a.out, model=a.model, mode=a.mode, max_new=max_new)


if __name__ == "__main__":
    main()
