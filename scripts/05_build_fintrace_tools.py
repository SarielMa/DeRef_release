"""FinTrace: tool catalog from the cached FMP docs and endpoint similarity.

Reads the FinTrace test set (evaluation/testset.json of HF `YupengCao/FinTrace`) and the cached docs pages. Writes
<work_dir>/fintrace/tools.jsonl and <work_dir>/fintrace/tool_sim.npz (BAAI/bge-large-en-v1.5 embeddings via
sentence-transformers on CPU; about a minute). The similarity only shapes the endpoint-substitution calls.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.build import fintrace  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", required=True, help="FinTrace evaluation/testset.json")
    ap.add_argument("--fmp-docs", required=True, help="directory with the cached docs pages (_index.json, <slug>.json)")
    ap.add_argument("--work-dir", required=True, help="build work directory")
    a = ap.parse_args()
    w = Path(a.work_dir) / "fintrace"
    fintrace.build_tools(Path(a.testset), Path(a.fmp_docs), w / "tools.jsonl")
    fintrace.tool_similarity(w / "tools.jsonl", w / "tool_sim.npz")


if __name__ == "__main__":
    main()
