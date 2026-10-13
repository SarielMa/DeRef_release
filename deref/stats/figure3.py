"""Figure 3: judge scores with the original DeRef reference vs the swapped reference, per dataset and judge.

Items: the swapped judge items (unit + negatives). Original score = the judge's DeRef score of the argument; swapped
score = its score on the swapped item. Metrics:
  correct_accept_0.5 / wrong_accept_0.5: share with score >= 0.5 among correct / wrong items, original, swapped and
      swapped minus original; cluster bootstrap over the query ids of those items (multinomial weights over their sorted
      clusters, B = 1,000, seed 20260930); se = SD (ddof 0) of the draws, se_delta = SD (ddof 1) of the difference draws;
  auroc: argument AUROC original / swapped and their difference with the paired `core.boot_auc` draws (se ddof 0,
      se_delta ddof 1).
lo / hi = 2.5 / 97.5 percentiles of the difference draws.
"""
import numpy as np
import pandas as pd

from deref.stats import core, load

DATASETS = ("cfb", "fcrb", "fintrace")
JUDGES = ("clm", "jev", "qwen3_8b", "qwen3_8_27b")
TAG = "deref_swap"


class ClusterMeans:
    """Cluster-bootstrap means: multinomial weights over the sorted clusters given (B x n, seed core.SEED)."""

    def __init__(self, clusters):
        self.cl = sorted(set(clusters))
        self.idx = {c: i for i, c in enumerate(self.cl)}
        n = len(self.cl)
        self.W = np.random.default_rng(core.SEED).multinomial(n, np.full(n, 1 / n), size=core.B).astype(float)

    def mean(self, x, cl):
        x = np.asarray(x, float)
        ci = np.array([self.idx[c] for c in cl])
        est = float(x.mean()) if len(x) else float("nan")
        num = self.W @ np.bincount(ci, weights=x, minlength=len(self.cl))
        den = self.W @ np.bincount(ci, minlength=len(self.cl)).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return est, num / den


def table(sets, scores_dir, swap_items, judges=JUDGES):
    """sets: {dataset: ArgSet}; swap_items: {dataset: [swapped judge rows in file order]}. Clusters are referred to by
    their position in the dataset's sorted cluster list, which orders them as the query ids themselves."""
    rows = []
    for d in DATASETS:
        items = swap_items[d]
        cidx = sets[d].cidx  # position of the query id in the sorted cluster list (same order as the ids)
        units = {r["unit"] for r in items}
        lb = np.array([r["label"] == "correct" for r in items], int)
        cl = [cidx[r["base_item"]] for r in items]
        for j in judges:
            plain = load.read_scores(f"{scores_dir}/{d}/plain/{j}.jsonl")
            deref = load.read_scores(f"{scores_dir}/{d}/deref/{j}.jsonl")
            sw = load.read_scores(f"{scores_dir}/{d}/{TAG}/{j}.jsonl")
            if plain is None or deref is None or sw is None:
                rows.append(dict(dataset=load.DISPLAY[d], judge=load.JUDGE_NAMES[j], units=len(units), metric="-", status="missing"))
                continue
            dr = load.over_plain(deref, "deref", plain)
            x = np.array([dr[r["base_item"]] for r in items], float)
            y = np.array([sw[r["item_id"]] for r in items], float)
            base = dict(dataset=load.DISPLAY[d], judge=load.JUDGE_NAMES[j], units=len(units))
            for m, name in ((lb == 1, "correct_accept_0.5"), (lb == 0, "wrong_accept_0.5")):
                clm = [c for c, f in zip(cl, m) if f]
                bt = ClusterMeans(clm)
                xa, xd = bt.mean((x[m] >= 0.5).astype(float), clm)
                ya, yd = bt.mean((y[m] >= 0.5).astype(float), clm)
                da, dd = bt.mean((y[m] >= 0.5).astype(float) - (x[m] >= 0.5), clm)
                lo, hi = np.nanpercentile(dd, [2.5, 97.5])
                rows.append(dict(**base, metric=name, n=int(m.sum()), orig=xa, orig_se=float(np.nanstd(xd)), swap=ya,
                                 swap_se=float(np.nanstd(yd)), delta=da, lo=lo, hi=hi, se_delta=float(np.nanstd(dd, ddof=1))))
            d_ = core.boot_auc(x, y, lb, np.array(cl))
            lo, hi = np.percentile(d_[:, 2], [2.5, 97.5])
            rows.append(dict(**base, metric="auroc", n=len(lb), orig=core.auc(x, lb), orig_se=float(np.std(d_[:, 0])),
                             swap=core.auc(y, lb), swap_se=float(np.std(d_[:, 1])), delta=core.auc(y, lb) - core.auc(x, lb),
                             lo=lo, hi=hi, se_delta=float(np.std(d_[:, 2], ddof=1))))
    return pd.DataFrame(rows)
