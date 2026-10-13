"""Figure 2b: acceptance at tau of the correct FC-RewardBench default values, with Plain, schema-facts and DeRef input.

Subsets of the correct arguments on the `default` path:
  reference_says_default: the DeRef reference of the argument contains the word "default" (case-insensitive, whole
      word) after Markdown emphasis markers are removed; an argument without a DeRef item has no reference (not counted);
  all_default: every correct argument on the `default` path.
tau and its tie share q are set per judge and input on all non-placeholder correct FC-RewardBench arguments (80%
accepted); the accepted share of a subset uses the dataset's cluster-bootstrap weights (`core.acc2`). Schema-facts
scores: an argument without a schema-facts item keeps its Plain score (as every other input). se = SD (ddof 0) of the
draws, lo / hi = 2.5 / 97.5 percentiles of the draws.
"""
import re

import numpy as np
import pandas as pd

from deref.inputs import expected_value_parser as EVP
from deref.stats import core, load

PREFIX = "Expected value (written before seeing the proposed value):\n"
JUDGES = ("clm", "qwen3_8b", "qwen3_8_27b", "jev")
CONDITIONS = (("plain", "Plain"), ("schema_facts", "Schema facts"), ("deref", "DeRef"))
SUBSETS = ("reference_says_default", "all_default")
INPUTS = ("plain", "schema_facts", "deref")
TAGS = {"schema_facts": "schema_facts", "deref": "deref"}


def reference_text(context):
    """The reference block of a DeRef judge input: text after PREFIX up to the next '\\nFunction: '."""
    i = context.index(PREFIX) + len(PREFIX)
    return context[i:context.index("\nFunction: ", i)]


def says_default(reference):
    return re.search(r"\bdefault\b", EVP.plain(EVP.normalise(reference)[0]), re.I) is not None


def reference_says_default(deref_items):
    """{argument item_id: bool} from the DeRef judge items (rows with base_item, context)."""
    return {r["base_item"]: says_default(reference_text(r["context"])) for r in deref_items}


def table(fc, scores_dir, deref_items, schema_items, judges=JUDGES):
    """fc: the FC-RewardBench ArgSet; deref_items / schema_items: the judge-input rows (for the subset and n_with_evidence)."""
    says = reference_says_default(deref_items)
    with_ev = {r["base_item"] for r in schema_items}
    cal = [k for k in fc.correct if fc.path[k] != "placeholder"]
    ci_cal = np.array([fc.cidx[k] for k in cal])
    sel = {"reference_says_default": lambda k: says.get(k, False), "all_default": lambda k: True}
    rows = []
    for j in judges:
        plain = load.read_scores(f"{scores_dir}/fcrb/plain/{j}.jsonl")
        assert plain is not None, f"missing Plain scores of {j}"
        for inp, cname in CONDITIONS:
            if inp == "plain":
                s = plain
            else:
                src = load.read_scores(f"{scores_dir}/fcrb/{inp}/{j}.jsonl")
                if src is None:
                    rows.append(dict(judge=load.JUDGE_NAMES[j], condition=cname, subset="-", status="missing"))
                    continue
                s = load.over_plain(src, TAGS[inp], plain)
            tau, q, tb, qb = core.tau_q_draws(load.arr(s, cal), ci_cal, fc.counts)
            for sub in SUBSETS:
                kk = [k for k in fc.correct if fc.path[k] == "default" and sel[sub](k)]
                ci_ = np.array([fc.cidx[k] for k in kk])
                a = core.acc2(load.arr(s, kk), ci_, tau, q, tb, qb, fc.counts)
                rows.append(dict(judge=load.JUDGE_NAMES[j], condition=cname, subset=sub, status="ok", n=len(kk),
                                 n_with_evidence=sum(k in with_ev for k in kk), tau=float(tau), q=float(q), acc=a.est,
                                 se=float(np.nanstd(a.boot)), lo=a.lo, hi=a.hi))
    return pd.DataFrame(rows)
