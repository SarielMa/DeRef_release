"""Table 2, self-generated reference rows (Qwen3-8B and Qwen3.8-27B): argument AUROC with 95% CI, and its paired
differences with Plain and DeRef, on all arguments and on each dataset's main out-of-view path (FC-RewardBench
default, ComplexFuncBench history_only, FinTrace no_source).

Reads the data package and `<scores>/<dataset>/{plain,deref,self_reference}/<judge>.jsonl`; self-reference items are
keyed `<argument item_id>|self_reference` (82_build_self_reference_items.py; every analysis argument has one).
CIs: query-clustered paired bootstrap (1,000 draws, seed 20260930), 2.5 / 97.5 percentiles; se = SD of the draws (ddof 1).
Writes <out-dir>/table2_self_reference.csv (proportions) and table2_self_reference_x100.csv.

  python scripts/86_table2_self_reference.py --data-dir <deref_data> --scores-dir <scores> --out-dir <tables>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.stats import core, load  # noqa: E402
from deref.stats.tables import times100  # noqa: E402

JUDGES = ("qwen3_8b", "qwen3_8_27b")
MAIN = {"fcrb": "default", "cfb": "history_only", "fintrace": "no_source"}
TAG = "self_reference"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--scores-dir", required=True, help="score files <scores_dir>/<dataset>/<input>/<judge>.jsonl")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    sets = load.load_sets(a.data_dir)
    S = Path(a.scores_dir)
    rows = []
    for d in load.DATASETS:
        st = sets[d]
        for j in JUDGES:
            plain = load.read_scores(S / d / "plain" / f"{j}.jsonl")
            own = load.read_scores(S / d / TAG / f"{j}.jsonl")
            deref = load.read_scores(S / d / "deref" / f"{j}.jsonl")
            keys_all = list(st.correct) + list(st.wrong)
            missing = [k for k in keys_all if f"{k}|{TAG}" not in own]
            assert not missing, f"{d} {j}: {len(missing)} arguments without a self-reference score"
            inp = {"self_reference": {k: own[f"{k}|{TAG}"] for k in keys_all}, "plain": plain,
                   "deref": load.over_plain(deref, load.TAG["deref"], plain)}
            for scope in ("all", MAIN[d]):
                kc = [k for k in st.correct if scope == "all" or st.path[k] == scope]
                kw = [k for k in st.wrong if scope == "all" or st.path[k] == scope]
                keys = kc + kw
                lb = np.r_[np.ones(len(kc)), np.zeros(len(kw))]
                cl = np.array([st.cidx[k] for k in keys])
                x = {n: load.arr(s, keys) for n, s in inp.items()}
                for n, v in x.items():
                    dr = core.boot_auc(v, v, lb, cl)
                    rows.append(dict(dataset=d, judge=j, kind="AUROC", input=n, scope=scope, n=len(keys),
                                     est=core.auc(v, lb), lo=float(np.percentile(dr[:, 0], 2.5)),
                                     hi=float(np.percentile(dr[:, 0], 97.5)), se=core.sd_draws(dr[:, 0])))
                for n in ("plain", "deref"):
                    dr = core.boot_auc(x[n], x["self_reference"], lb, cl)
                    rows.append(dict(dataset=d, judge=j, kind="paired difference", input=f"self_reference - {n}",
                                     scope=scope, n=len(keys), est=core.auc(x["self_reference"], lb) - core.auc(x[n], lb),
                                     lo=float(np.percentile(dr[:, 2], 2.5)), hi=float(np.percentile(dr[:, 2], 97.5)),
                                     se=core.sd_draws(dr[:, 2])))
    T = pd.DataFrame(rows)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "table2_self_reference.csv", index=False)
    X = times100(T, ["est", "lo", "hi", "se"])
    X.to_csv(out / "table2_self_reference_x100.csv", index=False)
    print(X[(X.scope == "all")].to_string(index=False))


if __name__ == "__main__":
    main()
