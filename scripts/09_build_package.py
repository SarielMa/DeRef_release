"""Derivation paths, the ComplexFuncBench parallel-call drop list, analysis sets and the data package.

Reads <work_dir>/states/... (steps 07-08), the per-dataset processed files, the FC-RewardBench source rows, the
ComplexFuncBench gold file and <work_dir>/fintrace/tools.jsonl. Writes <work_dir>/states/paths.jsonl,
<work_dir>/states/cfb_parallel_drop.json and the package files <out_dir>/{fcrb,cfb,fintrace}.jsonl and
<out_dir>/bootstrap_clusters.json. Needs `datasets`. CPU, a few minutes.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.build import package, paths  # noqa: E402
from deref.build.items import read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fcrb-raw", required=True, help="FC-RewardBench save_to_disk directory")
    ap.add_argument("--cfb-gold", required=True, help="ComplexFuncBench.jsonl")
    ap.add_argument("--work-dir", required=True, help="build work directory (after 08_build_states.py)")
    ap.add_argument("--out-dir", required=True, help="output directory of the data package files (data/)")
    a = ap.parse_args()
    w = Path(a.work_dir)
    sd = w / "states"
    state = {r["pos_id"]: r["state"] for d in ("fcrb", "cfb", "fintrace") for r in read_jsonl(sd / d / "positives.jsonl")}
    p = paths.census(sd / "export.jsonl", state, w / "fintrace" / "tools.jsonl")
    with open(sd / "paths.jsonl", "w", encoding="utf-8") as f:
        for k, v in p.items():
            f.write(json.dumps({"item_id": k, "path": v}) + "\n")
    cfb_calls = {c["call_id"]: c for c in read_jsonl(sd / "cfb" / "calls.jsonl")}
    drop = package.cfb_parallel_drop(sd / "export.jsonl", cfb_calls)
    with open(sd / "cfb_parallel_drop.json", "w", encoding="utf-8") as f:
        json.dump(drop, f, indent=1)
    print(f"[paths] {len(p)} arguments; [cfb parallel drop] {len(drop['drop_items'])} arguments, {len(drop['drop_calls'])} calls")
    package.write_package(w, Path(a.fcrb_raw), Path(a.cfb_gold), Path(a.out_dir))


if __name__ == "__main__":
    main()
