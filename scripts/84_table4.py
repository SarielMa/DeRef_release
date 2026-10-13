"""Table 4: reference availability x judge thinking on ComplexFuncBench with Qwen3.8-27B (argument AUROC).

Cells (all with the full history): Without = plain_full, With = deref_full; thinking off = the local judge
(`<scores>/cfb/<input>/qwen3_8_27b.jsonl`), thinking on = `<scores>/cfb/<input>/qwen3_8_27b_thinking.jsonl`
(83_score_local_judge_thinking.py). An argument without its own deref_full item keeps the Plain score of the same
thinking setting (`<scores>/cfb/plain/...`). Delta_ref = With - Without within a row; Delta_think = On - Off within a
column. CIs: 95% percentile intervals of the query-clustered paired bootstrap (1,000 draws, seed 20260930).
Writes <out-dir>/table4.csv (proportions) and table4_x100.csv.

  python scripts/84_table4.py --data-dir <deref_data> --scores-dir <scores> --out-dir <tables>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.stats import core, load  # noqa: E402
from deref.stats.tables import times100  # noqa: E402

D, J = "cfb", "qwen3_8_27b"
JUDGE = {"off": J, "on": f"{J}_thinking"}


def scores(scores_dir, think, inp):
    j = JUDGE[think]
    plain = load.read_scores(Path(scores_dir) / D / "plain" / f"{j}.jsonl")
    src = load.read_scores(Path(scores_dir) / D / inp / f"{j}.jsonl")
    assert plain is not None and src is not None, (think, inp)
    return load.over_plain(src, load.TAG[inp], plain)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--scores-dir", required=True, help="score files <scores_dir>/<dataset>/<input>/<judge>.jsonl")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    st = load.load_sets(a.data_dir)[D]
    keys = list(st.correct) + list(st.wrong)
    lb = np.r_[np.ones(len(st.correct)), np.zeros(len(st.wrong))]
    cl = np.array([st.cidx[k] for k in keys])
    X = {(t, i): load.arr(scores(a.scores_dir, t, i), keys) for t in ("off", "on") for i in ("plain_full", "deref_full")}
    name = {"plain_full": "without reference", "deref_full": "with reference"}
    rows = []
    for (t, i), x in X.items():
        dr = core.boot_auc(x, x, lb, cl)
        rows.append(dict(cell=f"thinking {t}, {name[i]}", kind="AUROC", est=core.auc(x, lb),
                         lo=float(np.percentile(dr[:, 0], 2.5)), hi=float(np.percentile(dr[:, 0], 97.5)),
                         se=float(np.std(dr[:, 0])), n_pos=int(lb.sum()), n_neg=int(len(lb) - lb.sum())))
    for label, (ta, ia), (tb, ib) in (
            ("Delta_ref, thinking off", ("off", "deref_full"), ("off", "plain_full")),
            ("Delta_ref, thinking on", ("on", "deref_full"), ("on", "plain_full")),
            ("Delta_think, without reference", ("on", "plain_full"), ("off", "plain_full")),
            ("Delta_think, with reference", ("on", "deref_full"), ("off", "deref_full"))):
        xa, xb = X[(ta, ia)], X[(tb, ib)]
        dr = core.boot_auc(xb, xa, lb, cl)
        rows.append(dict(cell=label, kind="paired difference", est=core.auc(xa, lb) - core.auc(xb, lb),
                         lo=float(np.percentile(dr[:, 2], 2.5)), hi=float(np.percentile(dr[:, 2], 97.5)),
                         se=core.sd_draws(dr[:, 2]), n_pos=int(lb.sum()), n_neg=int(len(lb) - lb.sum())))
    T = pd.DataFrame(rows)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "table4.csv", index=False)
    times100(T, ["est", "lo", "hi", "se"]).to_csv(out / "table4_x100.csv", index=False)
    print(times100(T, ["est", "lo", "hi", "se"]).to_string(index=False))


if __name__ == "__main__":
    main()
