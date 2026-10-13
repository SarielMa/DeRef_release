"""Analysis sets and the data package writer.

Argument-level analysis sets (per dataset): the wrong calls of the "parameter value" group (FC-RewardBench,
ComplexFuncBench) / "P4" group (FinTrace) that survived cleaning and the length rule (C2 main calls only), and their
correct calls. correct = every argument of those correct calls; wrong = every argument labelled wrong in those wrong
calls, minus (ComplexFuncBench) the parallel-call drop list (`cfb_parallel_drop`). The bootstrap cluster of an
argument is the query id of its state (FC-RewardBench test id, ComplexFuncBench task id, FinTrace query id).
"""
import json
import re
from collections import Counter, defaultdict

from deref.build import cfb as cfb_mod
from deref.build import fcrb as fcrb_mod
from deref.build.export import calls as parse_calls
from deref.build.items import read_jsonl
from deref.build.paths import PATH_NAMES

LONG = {"fcrb": "FC-RewardBench", "cfb": "ComplexFuncBench", "fintrace": "FinTrace"}
GROUP = {"fcrb": "parameter value", "cfb": "parameter value", "fintrace": "P4"}
REASON_UNCHANGED = "unchanged argument of a wrong call"
REASON_BFCL = "argument of a wrong call accepted by the BFCL checker"
REASON_V22 = "parallel same-name call aligned by position"
FT_TOOL_KEYS = ("endpoint", "category", "title", "description", "description_raw", "desc_source", "source_url", "params")


def gold_map(pos_c1, neg_c1):
    """{call index in the wrong step: arguments of the paired correct call} (pairing as in `export.wrong_items`)."""
    pcs, used, out = parse_calls(pos_c1), set(), {}
    for jn, (name, args) in enumerate(parse_calls(neg_c1)):
        i = next((i for i, (n, _) in enumerate(pcs) if n == name and i not in used), None)
        if i is not None:
            used.add(i)
        out[jn] = pcs[i][1] if i is not None else {}
    return out


def cfb_parallel_drop(export_path, cfb_calls):
    """ComplexFuncBench wrong arguments that are wrong only because parallel same-name calls were aligned by position:
    in a step with >= 2 correct calls of the same function, the value differs from its paired correct value but equals
    the same parameter (same type) of another same-name correct call. Wrong calls whose wrong arguments all lie in the
    list would be dropped at call level (none are). cfb_calls: {call_id: calls row} of the final ComplexFuncBench
    calls."""
    rows = []
    with open(export_path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["dataset"] != "ComplexFuncBench":
                continue
            posc = next(c for c in e["calls"] if c["role"] == "pos")
            for c in e["calls"]:
                if c["role"] != "neg":
                    continue
                gm = gold_map(posc["action_c1"], c["action_c1"])
                for it in c["items"]:
                    if it["kind"] == "param" and it["label"] == "wrong":
                        rows.append((it["item_id"], it["value"], gm.get(it["call_index"], {}).get(it["param"]),
                                     c["parsed"], posc["parsed"], it["label"]))
    label = {}
    n_multi, out = 0, []
    for k, value, gold, nparsed, pparsed, lab in sorted(rows, key=lambda r: r[0]):
        label[k] = lab
        cid, rest = k.split("|p|")
        j, p = rest.split(":", 1)
        name = nparsed[int(j)][0]
        same = [a for n, a, _ in pparsed if n == name]
        if len(same) < 2:
            continue
        n_multi += 1
        if value != gold and any(p in a and a[p] == value and type(a[p]) is type(value) for a in same):
            out.append(k)
    S = set(out)
    drop_calls, kept = [], []
    for cid in sorted({k.split("|p|")[0] for k in S}):
        w = [k for k in cfb_calls[cid]["param_items"] if label.get(k) == "wrong"]
        (drop_calls if set(w) <= S else kept).append({"call_id": cid, "wrong": len(w), "in_30": sum(k in S for k in w)})
    return {"n_multi": n_multi, "drop_items": sorted(out), "drop_calls": [c["call_id"] for c in drop_calls], "kept_calls": kept}


def analysis_sets(d, state_dir, labels, drop_items=()):
    """(correct argument ids, wrong argument ids) of one dataset, as sorted lists."""
    calls = {c["call_id"]: c for c in read_jsonl(state_dir / d / "calls.jsonl")}
    negs = [n for n in read_jsonl(state_dir / d / "negatives.jsonl") if n["group"] == GROUP[d]]
    correct = sorted({k for pid in dict.fromkeys(n["pos_id"] for n in negs) for k in calls[pid]["param_items"]})
    wrong = sorted({k for n in negs for k in calls[n["neg_id"]]["param_items"] if labels[k] == "wrong"})
    drop = set(drop_items)
    assert drop <= set(wrong)
    return correct, [k for k in wrong if k not in drop]


def rung(label, n_steps):
    """Ladder rung of a state as numbers (question cuts do not occur in the released data)."""
    if label == "none":
        return {"label": label, "drop_returns": 0, "drop_steps": 0, "question": None}
    m = re.fullmatch(r"returns dropped for (\d+) earliest step\(s\)", label)
    if m:
        return {"label": label, "drop_returns": int(m.group(1)), "drop_steps": 0, "question": None}
    m = re.fullmatch(r"all returns and (\d+) earliest step\(s\) dropped", label)
    if m:
        return {"label": label, "drop_returns": n_steps, "drop_steps": int(m.group(1)), "question": None}
    raise ValueError(f"unexpected rung label {label!r}")


def write_package(work, fcrb_raw, cfb_gold, out_dir):
    """-> out_dir/{fcrb,cfb,fintrace}.jsonl and out_dir/bootstrap_clusters.json (one record per candidate call)."""
    sd = work / "states"
    export_path = sd / "export.jsonl"
    with open(sd / "paths.jsonl", encoding="utf-8") as f:
        path = {r["item_id"]: r["path"] for r in map(json.loads, f)}
    with open(sd / "cfb_parallel_drop.json", encoding="utf-8") as f:
        drop30 = set(json.load(f)["drop_items"])
    labels = {}
    for d in LONG:
        labels.update({r["item_id"]: r["label"] for r in read_jsonl(sd / d / "labels.jsonl")})
    fc_tools = fcrb_mod.tool_catalogs(fcrb_raw)
    cfb_funcs = {tid: {f["name"]: f for f in r["functions"]} for tid, r in cfb_mod.read_gold(cfb_gold).items()}
    ft_tools = {t["endpoint"].lower(): {k: t[k] for k in FT_TOOL_KEYS if k in t}
                for t in read_jsonl(work / "fintrace" / "tools.jsonl")}
    proc_pos, proc_neg, vpos, clusters, sets = {}, {}, {}, {}, {}
    keep_negs = {d: defaultdict(list) for d in LONG}
    for d in LONG:
        pp = read_jsonl(work / d / "processed" / "positives.jsonl")
        proc_pos[d] = {p["pos_id"]: p for p in pp}
        proc_neg[d] = {n["neg_id"]: n for n in read_jsonl(work / d / "processed" / "negatives.jsonl")}
        clusters[d] = sorted({p["query_id"] for p in pp})
        vpos[d] = {p["pos_id"]: p for p in read_jsonl(sd / d / "positives.jsonl")}
        for n in read_jsonl(sd / d / "negatives.jsonl"):
            if n["group"] == GROUP[d]:
                keep_negs[d][n["pos_id"]].append(n["neg_id"])
        correct, wrong = analysis_sets(d, sd, labels, drop30 if d == "cfb" else ())
        sets[d] = set(correct) | set(wrong)
    recs = {d: [] for d in LONG}
    st = Counter()
    with open(export_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            d = {v: k for k, v in LONG.items()}[s["dataset"]]
            pid = s["pos_id"]
            if pid not in keep_negs[d]:
                continue
            nids = set(keep_negs[d][pid])
            history = []
            for (lab, txt), name in zip(s["history"][1:], s["history_names"][1:]):
                kind = lab.split(",", 1)[0]
                history.append({"label": lab, "kind": kind, "step": int(lab.rsplit("step ", 1)[1]), "name": name, "text": txt})
            window = {"header": s["header"], "steps": [[{"call": c, "return": r} for c, r in step] for step in s["prev"]]}
            prung = rung(vpos[d][pid]["truncation"], len(s["prev"]))
            pp = proc_pos[d][pid]
            for c in s["calls"]:
                if not (c["role"] == "pos" or c["call_id"] in nids):
                    continue
                cid, is_pos = c["call_id"], c["role"] == "pos"
                calls = [{"call_index": j, "name": n_, "arguments": a, "description": ds_} for j, (n_, a, ds_) in enumerate(c["parsed"])]
                funcs = {}
                for cl in calls:
                    n_ = cl["name"]
                    if n_ in funcs:
                        continue
                    if d == "fcrb":
                        funcs[n_] = fc_tools[pid][n_]
                    elif d == "cfb":
                        funcs[n_] = cfb_funcs[pp["task_id"]][n_]
                    else:
                        funcs[n_] = ft_tools[n_.lower()]
                drops = {it["item_id"] for it in c["items"] if it["kind"] == "drop"}
                args = []
                for it in c["items"]:
                    if it["kind"] != "param":
                        continue
                    k, lab = it["item_id"], it["label"]
                    assert lab in ("correct", "wrong") and lab == labels[k], k
                    did = k.replace("|p|", "|d|", 1)
                    assert did in drops, k
                    pth = PATH_NAMES[path[k]]
                    ina = k in sets[d]
                    why = None if ina else (REASON_V22 if lab == "wrong" else REASON_BFCL if d == "fcrb" else REASON_UNCHANGED)
                    assert ina == (is_pos or (lab == "wrong" and k not in drop30)), k
                    args.append({"item_id": k, "drop_item_id": did, "call_index": it["call_index"], "name": it["param"],
                                 "value": it["value"], "label": lab, "path": pth, "is_placeholder": pth == "placeholder",
                                 "in_analysis": ina, "exclusion_reason": why})
                    st[(d, lab, ina)] += 1
                if d == "fcrb":
                    src = proc_pos[d][pid] if is_pos else proc_neg[d][cid]
                    sm = {"test_category": src["test_category"]}
                    if not is_pos:
                        sm.update({"error_type": src["error_type"], "model_name": src["model_name"]})
                elif d == "cfb":
                    sm = {"task_id": pp["task_id"], "k": pp["k"]}
                    if not is_pos:
                        n = proc_neg[d][cid]
                        sm.update({"models": n["models"], "official_accepted": n["official_accepted"]})
                else:
                    sm = {"turn": pp["turn"], "call_idx": pp["call_idx"], "endpoint": pp["endpoint"]}
                    if not is_pos:
                        n = proc_neg[d][cid]
                        sm.update({"p4_subtype": n["p4_subtype"], "p4_change": n["p4_change"]})
                recs[d].append({
                    "dataset": d, "item_id": cid, "query_id": pp["query_id"], "state_id": pid,
                    "call_label": "correct" if is_pos else "wrong",
                    "request": s["question"], "history": history, "plain_window": window, "plain_rung": prung,
                    "action": c["action_c1"], "action_with_description": c["action_c2"], "calls": calls,
                    "functions": funcs, "arguments": args, "source_meta": sm})
                st[(d, "calls", "correct" if is_pos else "wrong")] += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in LONG:
        assert sum(v for (dd, lab, ina), v in st.items() if dd == d and ina is True) == len(sets[d]), d
        rows = sorted(recs[d], key=lambda r: (r["state_id"], r["call_label"] != "correct", r["item_id"]))
        with open(out_dir / f"{d}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "bootstrap_clusters.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({d: clusters[d] for d in LONG}, ensure_ascii=False) + "\n")
    for d in LONG:
        print(f"[package] {d}: calls correct {st[(d, 'calls', 'correct')]} / wrong {st[(d, 'calls', 'wrong')]}; "
              f"arguments in analysis correct {st[(d, 'correct', True)]} / wrong {st[(d, 'wrong', True)]}")
