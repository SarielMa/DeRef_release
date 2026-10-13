"""Result tables: Table 1 (Gap_tau with Plain input), the argument-level grid with paired differences (Table 2),
judge averages (Section 5.2) and the ablation (Table 3). All numbers are proportions."""
import numpy as np
import pandas as pd

from deref.stats import core
from deref.stats.load import (DATASETS, J3, J4, J7, JUDGE_NAMES, NONVIS, PATHS, ST_MISSING, ST_OK, arr, plain_scores)

# inputs of the argument grid (Table 2; deref_full for the full-history comparison)
GRID_IN = ("plain", "plain_full", "evidence_search", "revise", "deref", "deref_full")
# grid paths: the five paths, the four out-of-view paths pooled, and all arguments (placeholders included)
GRID_PATHS = PATHS + ("out_of_view", "all")
SUBSET = "full set (arguments without an own score keep the Plain score)"
GRID_COLS = ["dataset", "judge", "input", "path", "subset", "status", "n_pos", "n_neg", "tau", "q", "auroc", "auroc_se",
             "acc_c_tau", "acc_c_05", "acc_w_tau", "acc_w_05", "gap", "gap_se", "gap_lo", "gap_hi"]
DIFF_COLS = ["dataset", "judge", "pair", "path", "subset", "d_auroc", "d_auroc_lo", "d_auroc_hi", "d_gap", "d_gap_lo",
             "d_gap_hi", "se_d_auroc", "se_d_gap"]
POOLED_COLS = ["level", "dataset", "judge_set", "comparison", "method", "n_judges", "judges", "method_mean", "ref_mean",
               "mean_diff", "se", "lo", "hi", "wins", "ties", "losses", "n_draws"]
ABL_COLS = ["level", "ablation", "dataset", "judge", "ref", "x", "n_pos", "n_neg", "auroc_ref", "se_ref", "auroc_x", "se_x",
            "d", "lo", "hi", "se_d", "note"]
# ablation: (code, input X, reference, label); judge fixed to Qwen3.8-27B; D is skipped on FC-RewardBench
ABL = (("A", "sgv_prior", "deref", "w/o full history"),
       ("C", "deref_qwen3_8b", "deref", "Qwen3-8B as deriver"),
       ("F", "deref_nothink", "deref", "Deriver thinking off"),
       ("E", "expected_value", "deref", "Parsed expected value only"))
ABL_JUDGE = "qwen3_8_27b"
NOTE_SAME_DIFF = "same as argument_diffs"


def pair_name(a, b):
    return f"{a} - {b}"


def grid_pairs():
    """(X, plain) for X != plain; (deref, X) for X not in {deref, plain}; (deref_full, plain_full)."""
    pairs = [(x, "plain") for x in GRID_IN if x != "plain"]
    pairs += [("deref", x) for x in GRID_IN if x not in ("deref", "plain")]
    pairs += [("deref_full", "plain_full")]
    return pairs


def _mask(paths, P):
    if P == "out_of_view":
        return np.isin(paths, NONVIS)
    if P == "all":
        return np.ones(len(paths), bool)
    return paths == P


# ---------------------------------------------------------------- Table 1
def table1(sets, scores_dir, judges=J7):
    """Gap_tau of each judge's Plain scores: rows per dataset x judge x scope (out_of_view, then each out-of-view path
    with arguments). share = n / n(out_of_view); contrib = share * gap / gap(out_of_view)."""
    rows = []
    for d in DATASETS:
        st = sets[d]
        ck = list(st.correct)
        ci_ = np.array([st.cidx[k] for k in ck])
        pth = np.array([st.path[k] for k in ck])
        for j in judges:
            sc = plain_scores(scores_dir, d, j)
            if sc is None or not all(k in sc for k in ck):
                rows.append(dict(dataset=d, judge=j, scope=ST_MISSING))
                continue
            v = arr(sc, ck)
            res = core.gaps(v, pth, ci_, st.counts)
            knv = next(k for k in res if k[0] == "out_of_view")
            nv, n_nv = res[knv], knv[1]
            for (scope, n), g in res.items():
                rows.append(dict(dataset=d, judge=j, scope=scope, n=n, share=n / n_nv, gap=g.est,
                                 se=float(np.nanstd(g.boot)), lo=g.lo, hi=g.hi,
                                 contrib=(n / n_nv) * g.est / nv.est if scope != "out_of_view" else 1.0))
    return pd.DataFrame(rows, columns=["dataset", "judge", "scope", "n", "share", "gap", "se", "lo", "hi", "contrib"])


# ---------------------------------------------------------------- argument grid and paired differences
def grid(sets, SC, NOTE, inputs=GRID_IN, pairs=None, judges=J4, log=None):
    """Long table dataset x judge x input x path (n, tau, q, AUROC, accepted shares, Gap_tau) and the paired differences
    of `pairs` (AUROC difference with paired cluster-bootstrap CI; Gap_tau difference on the common weights).
    se_d_* are the standard deviations (ddof 1, NaN draws dropped) of the draws behind each CI."""
    pairs = grid_pairs() if pairs is None else pairs
    rows, drows = [], []
    for d in DATASETS:
        st = sets[d]
        ck, wk = list(st.correct), list(st.wrong)
        ci_c = np.array([st.cidx[k] for k in ck])
        pc_ = np.array([st.path[k] for k in ck])
        pw_ = np.array([st.path[k] for k in wk])
        clc = ci_c
        clw = np.array([st.cidx[k] for k in wk])
        cal = pc_ != "placeholder"
        for j in judges:
            if log:
                log(f"grid {d} {j}")
            G, A = {}, {}
            for inp in inputs:
                if inp not in SC[(d, j)]:
                    for P in GRID_PATHS:
                        rows.append(dict(dataset=d, judge=j, input=inp, path=P, status=NOTE.get((d, j, inp), ST_MISSING)))
                    continue
                s = SC[(d, j)][inp]
                vc = arr(s, ck)
                vw = arr(s, wk)
                tau, q, tb, qb = core.tau_q_draws(vc[cal], ci_c[cal], st.counts)
                G[inp] = {k[0]: v for k, v in core.gaps(vc, pc_, ci_c, st.counts).items()}
                for P in GRID_PATHS:
                    mc, mw = _mask(pc_, P), _mask(pw_, P)
                    scope = P if (P == "out_of_view" or P in NONVIS) else None
                    r = dict(dataset=d, judge=j, input=inp, path=P, status=NOTE.get((d, j, inp), ST_OK),
                             n_pos=int(mc.sum()), n_neg=int(mw.sum()), tau=float(tau), q=float(q))
                    if mc.sum() and mw.sum():
                        x = np.r_[vc[mc], vw[mw]]
                        lb = np.r_[np.ones(mc.sum()), np.zeros(mw.sum())]
                        cl = np.r_[clc[mc], clw[mw]]
                        dr = core.boot_auc(x, x, lb, cl)
                        r.update(auroc=core.auc(x, lb), auroc_se=float(np.std(dr[:, 0])))
                        A[(inp, P)] = (x, lb, cl)
                    if scope in G[inp]:
                        g = G[inp][scope]
                        r.update(gap=g.est, gap_se=float(np.nanstd(g.boot)), gap_lo=g.lo, gap_hi=g.hi)
                    if mc.sum():
                        r.update(acc_c_tau=core.acc2_point(vc[mc], tau, q), acc_c_05=float((vc[mc] >= 0.5).mean()))
                    if mw.sum():
                        r.update(acc_w_tau=core.acc2_point(vw[mw], tau, q), acc_w_05=float((vw[mw] >= 0.5).mean()))
                    rows.append(r)
            for a, b in pairs:
                if a not in SC[(d, j)] or b not in SC[(d, j)]:
                    continue
                if SC[(d, j)][a] is SC[(d, j)][b] or (d == "fcrb" and {a, b} in ({"plain_full", "plain"}, {"deref_full", "deref"})):
                    continue
                for P in GRID_PATHS:
                    r = dict(dataset=d, judge=j, pair=pair_name(a, b), path=P)
                    if (a, P) in A and (b, P) in A:
                        xa, lb, cl = A[(a, P)]
                        xb = A[(b, P)][0]
                        dr = core.boot_auc(xb, xa, lb, cl)
                        dd = core.auc(xa, lb) - core.auc(xb, lb)
                        lo, hi = np.percentile(dr[:, 2], [2.5, 97.5])
                        r.update(d_auroc=dd, d_auroc_lo=lo, d_auroc_hi=hi, se_d_auroc=core.sd_draws(dr[:, 2]))
                    scope = P if (P == "out_of_view" or P in NONVIS) else None
                    if scope and scope in G[a] and scope in G[b]:
                        g = G[a][scope] - G[b][scope]
                        r.update(d_gap=g.est, d_gap_lo=g.lo, d_gap_hi=g.hi, se_d_gap=core.sd_draws(g.boot))
                    if len(r) > 4:
                        drows.append(r)
    R = pd.DataFrame(rows)
    D = pd.DataFrame(drows)
    R.insert(4, "subset", SUBSET)
    D.insert(4, "subset", SUBSET)
    return R.reindex(columns=GRID_COLS), D.reindex(columns=DIFF_COLS)


# ---------------------------------------------------------------- judge averages (Table 2)
def param_draws(sets, SC, m, ref, judges=J4):
    """{("argument", d, j, m, ref): dict(est, lo, hi, a_m, a_ref, draws)} over all labelled arguments (all paths):
    AUROC(m) - AUROC(ref) with the paired cluster bootstrap."""
    G = {}
    for d in DATASETS:
        st = sets[d]
        ck, wk = list(st.correct), list(st.wrong)
        lb = np.r_[np.ones(len(ck)), np.zeros(len(wk))]
        cl = np.r_[[st.cidx[k] for k in ck], [st.cidx[k] for k in wk]]
        for j in judges:
            s = SC[(d, j)]
            if m not in s or ref not in s:
                continue
            xm = arr(s[m], ck + wk)
            xr = arr(s[ref], ck + wk)
            dr = core.boot_auc(xr, xm, lb, cl)
            lo, hi = np.percentile(dr[:, 2], [2.5, 97.5])
            G[("argument", d, j, m, ref)] = dict(est=core.auc(xm, lb) - core.auc(xr, lb), lo=float(lo), hi=float(hi),
                                                 a_m=core.auc(xm, lb), a_ref=core.auc(xr, lb), draws=dr[:, 2])
    return G


def comparisons():
    """(label, method, reference): every grid input except deref against deref, and deref against plain."""
    allin = [x for x in GRID_IN if x != "deref"]
    return [(pair_name(m, "deref"), m, "deref") for m in allin] + [(pair_name("deref", "plain"), "deref", "plain")]


def drop_fcrb_full(G):
    """plain_full / deref_full do not exist on FC-RewardBench (same as plain / deref)."""
    return {k: v for k, v in G.items() if not (k[1] == "fcrb" and ({"plain_full", "deref_full"} & {k[3], k[4]}))}


def pooled(G, comps, judges, label, levels=("argument",)):
    """Average over judges: per level x dataset x comparison, the draw-wise mean of the judges' bootstrap differences
    (resamples are shared across judges); CI = 2.5 / 97.5 percentiles of the averaged draws. wins / ties / losses count
    the judges whose own CI is above 0 / contains 0 / is below 0."""
    rows = []
    for lvl in levels:
        for d in DATASETS:
            for cname, m, ref in comps:
                g = [j for j in judges if (lvl, d, j, m, ref) in G]
                r = {"level": lvl, "dataset": d, "judge_set": label, "comparison": cname, "method": m}
                if not g:
                    rows.append({**r, "n_judges": 0, "judges": "not applicable"})
                    continue
                gg = [G[(lvl, d, j, m, ref)] for j in g]
                dr = np.mean(np.vstack([x["draws"] for x in gg]), axis=0)
                b = dr[~np.isnan(dr)]
                rows.append({**r, "n_judges": len(g), "judges": ";".join(g), "method_mean": np.mean([x["a_m"] for x in gg]),
                             "ref_mean": np.mean([x["a_ref"] for x in gg]), "mean_diff": np.mean([x["est"] for x in gg]),
                             "se": float(np.std(b, ddof=1)), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)),
                             "wins": sum(x["lo"] > 0 for x in gg), "ties": sum(x["lo"] <= 0 <= x["hi"] for x in gg),
                             "losses": sum(x["hi"] < 0 for x in gg), "n_draws": len(b)})
    return pd.DataFrame(rows).reindex(columns=POOLED_COLS)


def judge_averages(sets, SC):
    """Argument-level judge averages for all four judges ("all") and without Jev ("without_jev")."""
    comps = comparisons()
    G4 = {}
    for _, m, ref in comps:
        G4.update(param_draws(sets, SC, m, ref, J4))
    G4 = drop_fcrb_full(G4)
    G3 = {k: v for k, v in G4.items() if k[2] in J3}
    return pd.concat([pooled(G4, comps, J4, "all"), pooled(G3, comps, J3, "without_jev")], ignore_index=True)


# ---------------------------------------------------------------- ablation (Table 3)
def ablation(sets, SC, judge=ABL_JUDGE):
    """Each variant X against its reference on all labelled arguments (argument level, full set):
    d = AUROC(ref) - AUROC(X) with the paired cluster bootstrap; se_* = SD of the draws (ddof 0 for the AUROCs, 1 for d)."""
    recs = []
    for d in DATASETS:
        st = sets[d]
        ck, wk = list(st.correct), list(st.wrong)
        s = SC[(d, judge)]
        for ab, x, ref, nm in ABL:
            if x not in s or ref not in s:
                continue
            ks = ck + wk
            lb = np.r_[np.ones(len(ck)), np.zeros(len(wk))]
            cl = np.array([st.cidx[k] for k in ks])
            xr = arr(s[ref], ks)
            xx = arr(s[x], ks)
            dr = core.boot_auc(xx, xr, lb, cl)
            a_r, a_x = core.auc(xr, lb), core.auc(xx, lb)
            lo, hi = np.percentile(dr[:, 2], [2.5, 97.5])
            recs.append(dict(level="argument", ablation=nm, dataset=d, judge=judge, ref=ref, x=x, n_pos=len(ck), n_neg=len(wk),
                             auroc_ref=a_r, se_ref=float(np.std(dr[:, 1])), auroc_x=a_x, se_x=float(np.std(dr[:, 0])),
                             d=a_r - a_x, lo=float(lo), hi=float(hi), se_d=float(np.std(dr[:, 2], ddof=1)),
                             note=(NOTE_SAME_DIFF if ab == "A" else "")))
    return pd.DataFrame(recs).reindex(columns=ABL_COLS)


# ---------------------------------------------------------------- output helpers
def times100(df, cols):
    """Copy with the proportion columns `cols` multiplied by 100 and rounded to two decimals."""
    out = df.copy()
    for c in cols:
        if c in out:
            out[c] = (pd.to_numeric(out[c], errors="coerce") * 100).round(2)
    return out


def _ci(est, lo, hi):
    if pd.isna(est):
        return "-"
    return f"{est * 100:.2f} [{lo * 100:.2f}, {hi * 100:.2f}]"


MAIN_PATH = {"fcrb": "default", "cfb": "history_only", "fintrace": "no_source"}


def paper_table1(T1):
    """Table 1 view: per judge, per dataset Gap_tau x 100 [CI] out of view and on the dataset's main path."""
    rows = []
    for j in J7:
        r = {"judge": JUDGE_NAMES[j]}
        for d in DATASETS:
            for scope in ("out_of_view", MAIN_PATH[d]):
                x = T1[(T1.dataset == d) & (T1.judge == j) & (T1.scope == scope)]
                r[f"{d} {scope}"] = _ci(*x[["gap", "lo", "hi"]].iloc[0]) if len(x) else "-"
        rows.append(r)
    return pd.DataFrame(rows)


# Table 2 rows: (label, grid input or "self_reference")
TABLE2_ROWS = (("Plain", "plain"), ("Full history", "plain_full"), ("Self-generated reference", "self_reference"),
               ("ProvenanceGuard evidence", "evidence_search"), ("RevisEval", "revise"), ("DeRef", "deref"))


def paper_table2(G, D, SR=None):
    """Table 2 view: per judge, argument AUROC x 100 of each input on all arguments (grid `G`, path "all"; the
    self-generated reference from `SR` = table2_self_reference.csv if given), and Delta = deref - revise x 100 [CI]
    from the paired differences `D`. "-" = not run for that judge / dataset (FC-RewardBench has no cut, so no Full
    history row)."""
    G = G[G.path == "all"]
    D = D[(D.path == "all") & (D.pair == pair_name("deref", "revise"))]
    rows = []
    for j in J4:
        for label, inp in TABLE2_ROWS:
            r = {"judge": JUDGE_NAMES[j], "input": label}
            for d in DATASETS:
                if inp == "self_reference":
                    x = None if SR is None else SR[(SR.dataset == d) & (SR.judge == j) & (SR.kind == "AUROC")
                                                    & (SR.input == "self_reference") & (SR.scope == "all")]
                    v = x.est.iloc[0] if x is not None and len(x) else np.nan
                else:
                    x = G[(G.dataset == d) & (G.judge == j) & (G.input == inp) & (G.status == ST_OK)]
                    v = x.auroc.iloc[0] if len(x) else np.nan
                r[d] = "-" if pd.isna(v) else f"{v * 100:.2f}"
            if not all(r[d] == "-" for d in DATASETS):
                rows.append(r)
        r = {"judge": JUDGE_NAMES[j], "input": "Delta DeRef - RevisEval"}
        for d in DATASETS:
            x = D[(D.dataset == d) & (D.judge == j)]
            r[d] = _ci(*x[["d_auroc", "d_auroc_lo", "d_auroc_hi"]].iloc[0]) if len(x) else "-"
        rows.append(r)
    return pd.DataFrame(rows)


def paper_table3(A, dataset="cfb"):
    """Table 3 view (paper: ComplexFuncBench, Qwen3.8-27B): AUROC x 100 of deref and of each variant, and
    Delta = deref - variant x 100 [CI]."""
    A = A[A.dataset == dataset]
    rows = []
    if len(A):
        rows.append({"variant": "DeRef", "auroc": f"{A.auroc_ref.iloc[0] * 100:.2f}", "delta": ""})
    for _, x in A.iterrows():
        rows.append({"variant": x.ablation, "auroc": f"{x.auroc_x * 100:.2f}", "delta": _ci(x.d, x.lo, x.hi)})
    return pd.DataFrame(rows)
