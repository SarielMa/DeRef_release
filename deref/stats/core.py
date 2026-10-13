"""Core statistics: cluster bootstrap, AUROC, the acceptance threshold tau with its tie share q, and Gap_tau.

Bootstrap: clusters are query ids. B = 1,000 draws with seed 20260930.
- Gap_tau uses one fixed weight matrix per dataset: `cluster_counts(n_clusters)` draws multinomial cluster counts over
  the full sorted cluster list of the dataset (the order of `bootstrap_clusters.json`).
- AUROC and its paired differences use `boot_auc`, which resamples the clusters present in the items passed (fresh
  generator with the same seed on every call, so all judges and inputs on the same items share the same resamples).
CIs are 2.5 / 97.5 percentiles of the draws.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

B = 1000
SEED = 20260930
NONVIS = ("history_only", "format", "default", "no_source")  # the four out-of-view paths


@dataclass
class Stat:
    """Point estimate with its bootstrap draws (draws of two Stats on the same weights are paired)."""
    est: float
    boot: np.ndarray
    n: int = 0

    @property
    def lo(self):
        return float(np.nanpercentile(self.boot, 2.5))

    @property
    def hi(self):
        return float(np.nanpercentile(self.boot, 97.5))

    def __sub__(self, o):
        return Stat(self.est - o.est, self.boot - o.boot, min(self.n, o.n))

    def __neg__(self):
        return Stat(-self.est, -self.boot, self.n)


def cluster_counts(n_clusters, seed=SEED):
    """B x n_clusters multinomial cluster counts (each draw resamples n_clusters clusters uniformly)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=B).astype(float)


def sd_draws(draws):
    """Standard deviation (ddof 1) of the non-NaN draws; NaN if fewer than two."""
    b = np.asarray(draws, float)
    b = b[~np.isnan(b)]
    return float(np.std(b, ddof=1)) if len(b) > 1 else np.nan


# ---------------------------------------------------------------- AUROC
def auc(score, label):
    """AUROC by rank sum (ties get average ranks); label 1 = positive."""
    r = rankdata(score)
    n1 = label.sum()
    n0 = len(label) - n1
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_auc(rule, sc, label, clus):
    """Paired cluster bootstrap (B draws, seed SEED): rows (AUROC of rule, AUROC of sc, sc minus rule).
    Draws in which only one label class is present are skipped."""
    rng = np.random.default_rng(SEED)
    cl = np.unique(clus)
    idx = {c: np.flatnonzero(clus == c) for c in cl}
    out = []
    for _ in range(B):
        take = np.concatenate([idx[c] for c in rng.choice(cl, len(cl))])
        lb = label[take]
        if lb.min() == lb.max():
            continue
        a, b = auc(rule[take], lb), auc(sc[take], lb)
        out.append((a, b, b - a))
    return np.array(out)


# ---------------------------------------------------------------- tau, q and Gap_tau (tie rule v2)
def tau_q_draws(scores, ci, counts, qq=0.8):
    """tau, q (point) and per-draw tau_b, q_b.

    tau is the score of the ceil(0.8 n)-th correct argument in descending order; q is the share of the arguments tied at
    tau that is accepted so that exactly 80% are accepted. Per draw, the same with cluster weights."""
    scores = np.asarray(scores, float)
    n = len(scores)
    K = int(np.ceil(qq * n))
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    tau = s[K - 1]
    gt, eq = int((scores > tau).sum()), int((scores == tau).sum())
    q = (K - gt) / eq
    W = counts[:, ci[order]].astype(float)
    cum = np.cumsum(W, axis=1)
    tot = cum[:, -1]
    tau_b = s[(cum >= qq * tot[:, None]).argmax(axis=1)]
    Wgt = (W * (s[None, :] > tau_b[:, None])).sum(1)
    Weq = (W * (s[None, :] == tau_b[:, None])).sum(1)
    q_b = (qq * tot - Wgt) / Weq
    return tau, q, tau_b, q_b


def acc2(s, ci, tau, q, tau_b, q_b, counts):
    """Accepted share of the arguments s at (tau, q), with cluster-weighted bootstrap draws at (tau_b, q_b)."""
    s = np.asarray(s, float)
    if len(s) == 0:
        return None
    W = counts[:, ci].astype(float)
    a = (s[None, :] > tau_b[:, None]) + q_b[:, None] * (s[None, :] == tau_b[:, None])
    boot = (W * a).sum(1) / W.sum(1)
    est = float(((s > tau).sum() + q * (s == tau).sum()) / len(s))
    return Stat(est, boot, len(s))


def acc2_point(x, tau, q):
    """Point accepted share at (tau, q)."""
    x = np.asarray(x, float)
    return float(((x > tau).sum() + q * (x == tau).sum()) / len(x)) if len(x) else float("nan")


def gaps(v, path, ci_, counts):
    """Gap_tau of correct arguments: accepted share on `visible` minus accepted share on each out-of-view scope.

    v: scores of the correct arguments; path: their path labels; ci_: their cluster indices; tau is estimated on the
    non-placeholder correct arguments. Returns {(scope, n_scope): Stat} for scope `out_of_view` (the four out-of-view paths
    pooled) and each out-of-view path that has arguments, in that order."""
    cal = path != "placeholder"
    out = {}
    T2 = tau_q_draws(v[cal], ci_[cal], counts)
    a1 = np.flatnonzero(cal & (path == "visible"))
    for scope, sel in [("out_of_view", NONVIS)] + [(P, (P,)) for P in NONVIS]:
        aP = np.flatnonzero(np.isin(path, sel))
        if len(aP) == 0:
            continue
        out[(scope, len(aP))] = acc2(v[a1], ci_[a1], *T2, counts) - acc2(v[aP], ci_[aP], *T2, counts)
    return out
