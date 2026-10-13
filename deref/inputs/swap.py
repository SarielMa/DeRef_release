"""Reference swap (Figure 3): DeRef judge inputs in which the value the reference names is replaced by a wrong value.

Unit = a correct argument whose DeRef reference states a definite expected value (expected-value parser) equal to the gold
value (comparison `same_value`). Its argument group = the arguments sharing its derivation slot (state, Function line,
Parameter line, parameter name), i.e. the correct argument and the wrong arguments of the same slot; they share one
reference. Negatives of a unit: wrong arguments of the group that have a DeRef item, whose gold equals the unit's value
(ComplexFuncBench, FinTrace; FC-RewardBench wrong arguments are always paired with the unit) and whose own value
differs from the unit's value (`same_value`).
Gold of a wrong argument = value of the correct call's argument with the same call index and name at the same state, or,
if there is none, of the correct call's argument with the same name.
Per dataset (order ComplexFuncBench, FC-RewardBench, FinTrace; one generator `default_rng(20260930)` for all):
units with at least one negative are taken in argument-id order; if there are more than 500, 500 are drawn
(`rng.choice(n, 500, replace=False)`, kept in order). For each unit one negative is drawn (`rng.integers(n_neg)`, only
when it has more than one); its value replaces the parser's value span (`core_span`: the value text only, quotes and
emphasis markers around it kept). The swapped reference goes into the DeRef judge input of the unit and of every
negative (all other text unchanged). A unit is dropped if one of its swapped inputs exceeds 2,048 CLM tokens.
Item ids: swapped "<argument item_id>|deref_swap|<unit>" (a wrong argument can be a negative of two units of the same
slot, with two different swapped references); the original version is the DeRef item itself.
"""
import json
from collections import Counter

import numpy as np

from deref.generate import render as R
from deref.inputs import expected_value_parser as EVP
from deref.inputs import plain as PL
from deref.inputs import state as ST

PREFIX = "Expected value (written before seeing the proposed value):\n"
TAG = "deref_swap"
DATASETS = ("cfb", "fcrb", "fintrace")
MAX_UNITS = 500
SEED = 20260930
TRAIL = ".,;:"


def _text(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _norm(x):
    return " ".join(str(x).strip().strip("\"'`").lower().split())


def _norm_value(x):
    """Trailing . , ; : and surrounding quotes removed repeatedly, lower case, whitespace collapsed."""
    s = str(x)
    while True:
        t = s.strip().rstrip(TRAIL).strip().strip("\"'`")
        if t == s:
            break
        s = t
    if not s:
        return _norm(x)
    return " ".join(s.lower().split())


def same_value(read, value):
    """A value read from a reference (string) equals an argument value: equal after `_norm_value`, or equal as numbers."""
    a, b = _norm_value(read), _norm_value(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def reference_text(context):
    i = context.index(PREFIX) + len(PREFIX)
    return context[i:context.index("\nFunction: ", i)]


def _groups(records):
    """Analysis arguments of one dataset in record order, their slot, and the gold of wrong arguments."""
    args, slot, gold = [], {}, {}
    pos_by_suffix, pos_by_name = {}, {}
    for rec, a in ST.arguments(records, True):
        args.append((rec, a))
        _, fl, pl = R.split_main(PL.plain_item(rec, a)["context"])
        slot[a["item_id"]] = (rec["state_id"], fl, pl, a["name"])
        if rec["call_label"] == "correct":
            pos_by_suffix[(rec["state_id"], a["call_index"], a["name"])] = a["value"]
            pos_by_name.setdefault((rec["state_id"], a["name"]), []).append(a["value"])
    for rec, a in args:
        if rec["call_label"] == "wrong":
            k = (rec["state_id"], a["call_index"], a["name"])
            if k in pos_by_suffix:
                gold[a["item_id"]] = pos_by_suffix[k]
            else:
                vs = pos_by_name.get((rec["state_id"], a["name"]), [])
                gold[a["item_id"]] = vs[0] if vs else None
    return args, slot, gold


def build(records_by_dataset, deref_by_dataset, tok, log=print):
    """records_by_dataset {dataset: records}; deref_by_dataset {dataset: {argument item_id: DeRef judge row}}.
    -> (units, swapped rows, original rows, stats)."""
    rng = np.random.default_rng(SEED)
    st = Counter()
    units, swap_rows, orig_rows = [], [], []
    for d in DATASETS:
        step2 = deref_by_dataset[d]
        args, slot, gold = _groups(records_by_dataset[d])
        by_slot = {}
        for rec, a in args:
            by_slot.setdefault(slot[a["item_id"]], []).append(a)
        cand = []
        for rec, x in sorted(args, key=lambda t: t[1]["item_id"]):
            if x["label"] != "correct" or x["item_id"] not in step2:
                continue
            ref = reference_text(step2[x["item_id"]]["context"])
            r = EVP.prior_value12(ref)
            if not (r["definite"] and same_value(r["value"], x["value"])):
                continue
            st[f"{d}: eligible units (definite, reference = gold)"] += 1
            negs = []
            for w in by_slot[slot[x["item_id"]]]:
                if w["label"] != "wrong" or w["item_id"] not in step2:
                    continue
                if d != "fcrb" and gold[w["item_id"]] != x["value"]:
                    continue
                if same_value(_text(w["value"]), x["value"]):
                    continue
                negs.append(w)
            if not negs:
                st[f"{d}: dropped, no negative"] += 1
                continue
            cand.append((x, negs, ref, r))
        st[f"{d}: units with negatives"] = len(cand)
        if len(cand) > MAX_UNITS:
            idx = sorted(rng.choice(len(cand), MAX_UNITS, replace=False))
            cand = [cand[i] for i in idx]
        for x, negs, ref, r in cand:
            w0 = negs[int(rng.integers(len(negs)))] if len(negs) > 1 else negs[0]
            a, b = r["core_span"]
            swapped = ref[:a] + _text(w0["value"]) + ref[b:]
            assert EVP.plain(EVP.normalise(ref[a:b])[0]) == r["core"], x["item_id"]
            st[f"{d}: swapped references with the gold string still elsewhere"] += _text(x["value"]) in ref[:a] + ref[b:]
            rows, ok = [], True
            for it in [x] + negs:
                rr = step2[it["item_id"]]
                assert rr["context"].count(PREFIX + ref) == 1, it["item_id"]
                ctx_s = rr["context"].replace(PREFIX + ref, PREFIX + swapped, 1)
                n = ST.clm_tokens(tok, ctx_s, rr["question"])
                if n > ST.LIMIT:
                    ok = False
                    break
                rows.append((it, rr, ctx_s, n))
            if not ok:
                st[f"{d}: dropped, swapped item over 2048"] += 1
                continue
            units.append({"unit": x["item_id"], "dataset": d, "gold": x["value"], "swap_value": w0["value"],
                          "swap_from": w0["item_id"], "negatives": [w["item_id"] for w in negs], "reference": ref,
                          "reference_swapped": swapped, "value_span": [a, b], "value_as_read": r["text"]})
            for it, rr, ctx_s, n in rows:
                orig_rows.append({"item_id": rr["item_id"], "base_item": it["item_id"], "unit": x["item_id"], "dataset": d,
                                  "version": "original", "label": it["label"], "context": rr["context"],
                                  "question": rr["question"]})
                swap_rows.append({"item_id": f"{it['item_id']}|{TAG}|{x['item_id']}", "base_item": it["item_id"], "unit": x["item_id"],
                                  "dataset": d, "version": "swap", "label": it["label"], "context": ctx_s,
                                  "question": rr["question"], "clm_tokens": n})
        st[f"{d}: units sampled and built"] = sum(u["dataset"] == d for u in units)
        st[f"{d}: swapped items"] = sum(r["dataset"] == d for r in swap_rows)
        log(f"[swap] {d}: {st[f'{d}: units sampled and built']} units, {st[f'{d}: swapped items']} swapped items")
    assert len({r["item_id"] for r in swap_rows}) == len(swap_rows)
    by_unit = {u["unit"]: u for u in units}
    for o, s in zip(orig_rows, swap_rows):  # swapping back gives the DeRef input byte for byte
        u = by_unit[s["unit"]]
        assert o["base_item"] == s["base_item"]
        assert s["context"].replace(PREFIX + u["reference_swapped"], PREFIX + u["reference"], 1) == o["context"]
    return units, swap_rows, orig_rows, dict(st)
