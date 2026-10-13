"""Figure 2a: argument AUROC on FCRB-Relocate per version, judge and input.

Items: the relocated argument of every record of a version; positives = role `pos` (correct value), negatives = roles
`H` (the distractor in view) and `C` (a value that appears nowhere in the input). Clusters = source FC-RewardBench
state (`query_id`). Scores: `plain` is the base; `deref` items are `<argument item_id>|deref` and `plain_full` items
the bare argument id; an argument without its own score keeps its Plain score. CLM has no full-history input.
AUROC with its cluster-bootstrap SD (ddof 0) and 2.5 / 97.5 percentile interval (`core.boot_auc`)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from deref.stats import core
from deref.stats.load import JUDGE_NAMES, J4, NO_FULL_TEXT, read_scores

DATASET = "fcrb_relocate"
INPUTS = ("plain", "plain_full", "deref")
TAG = {"plain": "", "plain_full": "", "deref": "deref"}
VERSIONS = ("visible", "buried")


def load_items(data_dir):
    """[(argument item id, version, label 1/0, cluster)] of the relocated arguments."""
    d = Path(data_dir)
    p = d / "data" / f"{DATASET}.jsonl"
    p = p if p.exists() else d / f"{DATASET}.jsonl"
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                for a in r["arguments"]:
                    if a["in_analysis"]:
                        out.append((a["item_id"], r["version"], int(r["role"] == "pos"), r["query_id"]))
    return out


def figure2a(data_dir, scores_dir, judges=J4):
    items = load_items(data_dir)
    rows = []
    for ver in VERSIONS:
        its = [x for x in items if x[1] == ver]
        keys = [x[0] for x in its]
        lb = np.array([x[2] for x in its])
        cl = np.array([x[3] for x in its])
        for j in judges:
            plain = read_scores(Path(scores_dir) / DATASET / "plain" / f"{j}.jsonl")
            for inp in INPUTS:
                base = dict(version=ver, judge=JUDGE_NAMES[j], input=inp)
                if inp == "plain_full" and j in NO_FULL_TEXT:
                    rows.append({**base, "status": "not applicable"})
                    continue
                src = plain if inp == "plain" else read_scores(Path(scores_dir) / DATASET / inp / f"{j}.jsonl")
                if plain is None or src is None:
                    rows.append({**base, "status": "missing"})
                    continue
                suf = f"|{TAG[inp]}" if TAG[inp] else ""
                x = np.array([src.get(k + suf, plain[k]) for k in keys], float)
                dr = core.boot_auc(x, x, lb, cl)
                rows.append({**base, "status": "ok", "n_pos": int(lb.sum()), "n_neg": int((1 - lb).sum()),
                             "auroc": core.auc(x, lb), "se": float(np.std(dr[:, 0])),
                             "lo": float(np.nanpercentile(dr[:, 0], 2.5)), "hi": float(np.nanpercentile(dr[:, 0], 97.5))})
    return pd.DataFrame(rows)
