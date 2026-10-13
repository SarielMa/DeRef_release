"""ComplexFuncBench: model trajectories -> first-divergence steps -> correct / wrong step pairs -> per-argument inputs.

Sources
  gold:          HF `zai-org/ComplexFuncBench`, file ComplexFuncBench.jsonl (fields id, functions, conversations).
                 The HF revision was not recorded; the file is pinned by CFB_GOLD_SHA256.
  trajectories:  HF `AgentSuite/complex-func-bench-trajectories` at revision TRAJ_REVISION, one <model>.jsonl per model
                 (fields meta.id, messages[]).
  audit lists:   AgentSuite `cfbench_filter_ids.csv` (129 task ids) and `cfbench_issues.csv` (176 task ids). Their
                 origin (repository / revision) was not recorded; they are given by path.

Positive = gold step k; negative = a model's step k (first divergence from gold, deduplicated); both share the gold
history. Tokenizer for the length rule: Qwen/Qwen3-8B at QWEN3_8B_REVISION (CPU, tokenizer only).
"""
import collections
import csv
import json
import re
from datetime import datetime

from deref.build.items import NOUL_Q, PAIR_Q, Q_VALUE, NO_DESC, noul_context, read_jsonl

TRAJ_REVISION = "fcc906da345ec46d8815792bfff4724d7cca86b0"
CFB_GOLD_SHA256 = "be1e0f5951b666f6543b946e915a431a39bbda1807481878eb331245276ac088"
QWEN3_8B = "Qwen/Qwen3-8B"
QWEN3_8B_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

# fixed reply of the benchmark's API simulator to a call it cannot answer
UNEXPECTED = {"api_status": True, "content": "There is a problem with your api call, please double-check for possible problems."}
ORDER = ["function name", "missing params", "extra params", "missing + extra params", "param value"]
DATE_FMTS = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%Y%m%d"]
LOC = {"Location_to_Lat_Long", "Search_Hotel_Destination", "Search_Attraction_Location", "Search_Car_Location",
       "Search_Flight_Location", "Taxi_Search_Location"}
CUT, LAST, LIMIT = 1000, 3, 2048


def read_gold(gold_path):
    """{task id: source row} of ComplexFuncBench.jsonl, in file order."""
    return {r["id"]: r for r in read_jsonl(gold_path)}


def audit_lists(agentsuite_dir):
    """(filter ids, {task id: [detection, ...]}) from cfbench_filter_ids.csv and cfbench_issues.csv."""
    with open(agentsuite_dir / "cfbench_filter_ids.csv", encoding="utf-8") as f:
        flagged = {row["task_id"] for row in csv.DictReader(f)}
    issues = collections.defaultdict(list)
    with open(agentsuite_dir / "cfbench_issues.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            issues[row["task_id"]].append(row["detection"])
    return flagged, issues


# ---------------------------------------------------------------- step 1: first divergence
def norm(v):
    """Normalised value: strings lower-cased / whitespace collapsed; numbers -> float; times -> HH:MM; dates -> ISO."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {str(k): norm(x) for k, x in v.items()}
    s = re.sub(r"\s+", " ", str(v)).strip().lower()
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
        return round(float(s), 6)
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(:\d{2})?", s)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    for f in DATE_FMTS:
        try:
            return datetime.strptime(s, f).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return s


def ckey(c):
    return json.dumps([c["name"], norm(c.get("arguments") or {})], sort_keys=True)


def classify(pred, gold):
    """(category or None if equal, several differing pairs, [(pred call, gold call)])."""
    if len(pred) != len(gold):
        return "call count differs", False, []
    if sorted(map(ckey, pred)) == sorted(map(ckey, gold)):
        return None, False, []
    # pair identical calls away, then same names, then the rest in order
    p, g = list(pred), list(gold)
    for c in list(p):
        k = ckey(c)
        hit = next((x for x in g if ckey(x) == k), None)
        if hit is not None:
            p.remove(c), g.remove(hit)
    pairs = []
    for c in list(p):
        hit = next((x for x in g if x["name"] == c["name"]), None)
        if hit is not None:
            pairs.append((c, hit)), p.remove(c), g.remove(hit)
    pairs += list(zip(p, g))
    kinds = []
    for c, x in pairs:
        if c["name"] != x["name"]:
            kinds.append("function name")
            continue
        a, b = norm(c.get("arguments") or {}), norm(x.get("arguments") or {})
        miss, extra = set(b) - set(a), set(a) - set(b)
        if miss and extra:
            kinds.append("missing + extra params")
        elif miss:
            kinds.append("missing params")
        elif extra:
            kinds.append("extra params")
        else:
            kinds.append("param value")
    cat = min(kinds, key=ORDER.index)
    return cat, len(kinds) > 1, [(c, x) for c, x in pairs]


def gold_steps(conv):
    """(steps, observations): the function_call list of every assistant turn and the following observation."""
    steps, obs = [], []
    for i, m in enumerate(conv):
        if m["role"] == "assistant" and "function_call" in m:
            steps.append(m["function_call"])
            obs.append(conv[i + 1]["content"] if i + 1 < len(conv) and conv[i + 1]["role"] == "observation" else None)
    return steps, obs


def first_divergence(gold_path, traj_dir, agentsuite_dir, out_path):
    """Per model trajectory: the first assistant turn whose calls differ from the gold step (given equal earlier turns)."""
    gold = {}
    for tid, r in read_gold(gold_path).items():
        st, ob = gold_steps(r["conversations"])
        gold[tid] = {"q": r["conversations"][0]["content"], "steps": st, "obs": ob,
                     "nparams": [max((len(c["arguments"]) for c in s), default=0) for s in st]}
    flagged, issues = audit_lists(agentsuite_dir)
    rows = []
    for mdl in sorted(p.stem for p in traj_dir.glob("*.jsonl")):
        for t in read_jsonl(traj_dir / f"{mdl}.jsonl"):
            tid = t["meta"]["id"]
            g = gold[tid]
            msgs = t["messages"]
            k = 0
            for i, m in [(i, m) for i, m in enumerate(msgs) if m["role"] == "assistant"]:
                if k >= len(g["steps"]):
                    break
                pred = [{"name": c["name"], "arguments": c.get("arguments") or {}} for c in (m.get("tool_calls") or [])]
                gs = g["steps"][k]
                if not pred:
                    cat, multi, pairs = "no call (text answer)", False, []
                else:
                    cat, multi, pairs = classify(pred, gs)
                if cat is None:
                    k += 1
                    continue
                nxt = msgs[i + 1] if i + 1 < len(msgs) and msgs[i + 1]["role"] == "observation" else None
                if not pred:
                    acc = "n/a"
                elif nxt is None:
                    acc = "none"
                else:
                    ok = [not (o == UNEXPECTED or (isinstance(o, dict) and "error" in o)) for o in nxt["content"]]
                    acc = "all" if all(ok) else ("none" if not any(ok) else "partial")
                rows.append({"model": mdl, "task_id": tid, "domain": tid.rsplit("-", 1)[0], "k": k + 1, "category": cat,
                             "multi": multi, "n_calls_gold": len(gs), "n_calls_pred": len(pred),
                             "n_params_gold": g["nparams"][k], "gold": gs, "pred": pred,
                             "pairs": [{"pred": c, "gold": x} for c, x in pairs], "official_accepted": acc,
                             "agentsuite_filtered": tid in flagged, "agentsuite_issues": issues.get(tid, [])})
                break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"[cfb first divergence] {len(rows)} rows")


# ---------------------------------------------------------------- step 2: main sample (deterministic selection)
def leaves(v):
    if isinstance(v, dict):
        for x in v.values():
            yield from leaves(x)
    elif isinstance(v, list):
        for x in v:
            yield from leaves(x)
    else:
        yield v


def found(v, text):
    """Value visibility: strings case-insensitive substring, numbers exact digits; None for booleans / null."""
    if isinstance(v, bool) or v is None:
        return None  # not checked
    if isinstance(v, (int, float)):
        return re.search(rf"(?<![\d.]){re.escape(repr(v) if isinstance(v, float) else str(v))}(?![\d])", text) is not None
    return str(v).lower() in text.lower()


def main_sample(pairs_path, out_path):
    """Keep function-name / parameter-value divergences rejected by the official evaluator, not in the AgentSuite
    filter list and not location look-ups only; deduplicate by (task, step, normalised model calls)."""
    rows = read_jsonl(pairs_path)

    def loc_only(r):
        """Every differing pair involves a location look-up on the gold or the model side."""
        return all(p["gold"]["name"] in LOC or p["pred"]["name"] in LOC for p in r["pairs"])

    def keep(r):
        return (r["category"] in ("function name", "param value") and r["official_accepted"] != "all"
                and not r["agentsuite_filtered"] and not loc_only(r))
    uniq = {}
    for r in (r for r in rows if keep(r)):
        key = (r["task_id"], r["k"], json.dumps(sorted(map(ckey, r["pred"]))))
        if key not in uniq:
            uniq[key] = {**{k: v for k, v in r.items() if k != "model"}, "models": []}
        uniq[key]["models"].append(r["model"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in uniq.values())
    print(f"[cfb main sample] {len(uniq)} rows")


# ---------------------------------------------------------------- step 4 helpers: per-argument inputs
def step_action(calls, desc):
    return "\n".join(f"call {c['name']}({json.dumps(c.get('arguments') or {})}) — {desc[c['name']]}" for c in calls)


def calls_of(action_c1):
    out = []
    for line in action_c1.split("\n"):
        name, raw = line[len("call "):].split("({", 1)  # function names can contain "(" (e.g. Get_Hotel_Reviews(Tips)_Sort_Option)
        out.append({"name": name, "arguments": json.loads("{" + raw[:-1])})
    return out


def step_items(state, action_c1, fs, cid=""):
    """[(item_id, context, question)] of the argument checks and drop variants of one step under `state`.
    Item ids carry the call index inside the step: `<call_id>|p|<j>:<param>`, `<call_id>|d|<j>:<param>`."""
    desc = {k: v["description"] for k, v in fs.items()}
    step = calls_of(action_c1)
    out = []
    for j, call in enumerate(step):
        props = (fs[call["name"]].get("parameters") or {}).get("properties") or {}
        for k, v in call["arguments"].items():
            info = (props.get(k) or {}).get("description") or NO_DESC
            out.append((f"{cid}|p|{j}:{k}", f"{state}\nFunction: {call['name']} — {desc[call['name']]}\nParameter: {k} — {info}\n"
                        f"Proposed value for {k}: {json.dumps(v, ensure_ascii=False)}", Q_VALUE))
            dropped = [dict(cc, arguments={kk: vv for kk, vv in cc["arguments"].items() if kk != k}) if jj == j else cc
                       for jj, cc in enumerate(step)]
            out.append((f"{cid}|d|{j}:{k}", noul_context(state, step_action(dropped, desc)), NOUL_Q))
    return out


# ---------------------------------------------------------------- step 3: samples with the length rule
def build_state(q, steps, obs, k, drop_ret=0, drop_steps=0):
    """Question + the last 3 gold steps before k (calls, returns cut to 1,000 characters).
    drop_ret: returns removed for the earliest `drop_ret` shown steps; drop_steps: earliest shown steps removed."""
    lines = []
    shown = list(range(max(0, k - 1 - LAST), k - 1))[drop_steps:]
    for t, i in enumerate(shown):
        for j, c in enumerate(steps[i]):
            lines.append(f"- {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})")
            o = obs[i][j] if obs[i] and j < len(obs[i]) else None
            if o is not None and t >= drop_ret:
                lines.append(f"  -> {json.dumps(o, ensure_ascii=False)[:CUT]}")
    return f"Question: {q}\nPrevious calls (most recent {LAST} steps):\n" + ("\n".join(lines) if lines else "none")


def actions(calls, desc):
    c1 = "\n".join(f"call {c['name']}({json.dumps(c.get('arguments') or {})})" for c in calls)
    c2 = None if any(c["name"] not in desc for c in calls) else \
        "\n".join(f"call {c['name']}({json.dumps(c.get('arguments') or {})}) — {desc[c['name']]}" for c in calls)
    return c1, c2


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN3_8B, revision=QWEN3_8B_REVISION)


def build_samples(gold_path, main_path, out_dir, tok=None):
    """Positives / negatives with the shared state of each step.

    Length rule (same truncated state for a positive and all its negatives): the longest CLM input built from the
    state (whole-call pointwise C1 / C2 = state + call + question, pairwise / native = state + question, and the
    per-argument checks and drop variants of the positive and every negative) must be <= 2048 Qwen3-8B tokens
    (no special tokens). Otherwise, in order: drop the returns of the earliest shown step (repeat), then drop the
    earliest shown steps, then cut the question in the middle (" … ", >= 40 tokens kept, steps of 16). States still
    too long -> excluded_too_long.jsonl.
    Value-visibility rule: a truncated state where some leaf value of the gold step's arguments is found before but
    not after truncation is removed with all its negatives (excluded_value_lost.jsonl)."""
    gold = {}
    funcs = {}
    for tid, r in read_gold(gold_path).items():
        st, ob = gold_steps(r["conversations"])
        gold[tid] = {"q": r["conversations"][0]["content"], "steps": st, "obs": ob,
                     "desc": {f["name"]: f["description"] for f in r["functions"]},
                     "pdesc": {f["name"]: {p: v.get("description") for p, v in f["parameters"].get("properties", {}).items()}
                               for f in r["functions"]}}
        funcs[tid] = {f["name"]: f for f in r["functions"]}
    main_rows = read_jsonl(main_path)
    pos, neg = {}, []
    cat = {"function name": "func_name", "param value": "param_value"}
    for r in sorted(main_rows, key=lambda x: (x["task_id"], x["k"], x["category"], json.dumps(x["pred"]))):
        g, k = gold[r["task_id"]], r["k"]
        pid = f"{r['task_id']}|k{k}"
        if pid not in pos:
            gs = g["steps"][k - 1]
            a1, a2 = actions(gs, g["desc"])
            assert a2 is not None
            pos[pid] = {"pos_id": pid, "query_id": r["task_id"], "task_id": r["task_id"], "domain": r["domain"], "k": k,
                        "n_calls": len(gs), "n_params": r["n_params_gold"], "endpoints": [c["name"] for c in gs],
                        "state": build_state(g["q"], g["steps"], g["obs"], k), "action_c1": a1, "action_c2": a2,
                        "param_desc": {c["name"]: g["pdesc"].get(c["name"], {}) for c in gs}}
        n1, n2 = actions(r["pred"], g["desc"])
        idx = sum(n["pos_id"] == pid for n in neg) + 1
        neg.append({"neg_id": f"{pid}|n{idx}", "pos_id": pid, "query_id": r["task_id"], "perturb_type": cat[r["category"]],
                    "models": r["models"], "n_models": len(r["models"]), "official_accepted": r["official_accepted"],
                    "multi": r["multi"], "n_calls": len(r["pred"]), "endpoints": [c["name"] for c in r["pred"]],
                    "action_c1": n1, "action_c2": n2})

    # ---- length rule (Qwen3-8B tokenizer)
    import numpy as np
    tok = tok or load_tokenizer()
    ntok = lambda xs: np.array([len(i) for i in tok(list(xs), add_special_tokens=False)["input_ids"]])  # noqa: E731
    negs_of = collections.defaultdict(list)
    for n in neg:
        negs_of[n["pos_id"]].append(n)

    def inputs(state, p):
        acts = [(p["action_c1"], [n["action_c1"] for n in negs_of[p["pos_id"]]]),
                (p["action_c2"], [n["action_c2"] for n in negs_of[p["pos_id"]] if n["action_c2"] is not None])]
        xs = [state, f"{state}\n\n{PAIR_Q}"]
        fs = funcs[p["task_id"]]
        for a1 in [p["action_c1"]] + [n["action_c1"] for n in negs_of[p["pos_id"]]]:
            xs += [f"{ctx}\n\n{q}" for _, ctx, q in step_items(state, a1, fs)]
        for pa, nas in acts:
            for a in [pa] + nas:
                xs.append(f"{noul_context(state, a)}\n\n{NOUL_Q}")
        return xs

    def longest(state, p):
        return int(ntok(inputs(state, p)).max())

    def cut_question(q, keep):
        ids = tok(q, add_special_tokens=False)["input_ids"]
        if len(ids) <= keep:
            return q
        h = keep // 2
        return tok.decode(ids[:h]) + " … " + tok.decode(ids[-(keep - h):])

    excluded = []
    for pid, p in pos.items():
        g, k = gold[p["task_id"]], p["k"]
        shown = len(range(max(0, k - 1 - LAST), k - 1))
        info = {"returns_dropped_steps": 0, "previous_steps_dropped": 0, "question_cut_to_tokens": None}
        L0 = longest(p["state"], p)
        state, L = p["state"], L0
        for d in range(1, shown + 1):
            if L <= LIMIT:
                break
            state = build_state(g["q"], g["steps"], g["obs"], k, drop_ret=d)
            L, info["returns_dropped_steps"] = longest(state, p), d
        for ds in range(1, shown + 1):
            if L <= LIMIT:
                break
            state = build_state(g["q"], g["steps"], g["obs"], k, drop_ret=shown, drop_steps=ds)
            L, info["previous_steps_dropped"] = longest(state, p), ds
        if L > LIMIT:
            nq = len(tok(g["q"], add_special_tokens=False)["input_ids"])
            for keep in range(nq - 16, 39, -16):
                q2 = cut_question(g["q"], keep)
                state = build_state(q2, g["steps"], g["obs"], k, drop_ret=shown, drop_steps=shown)
                L, info["question_cut_to_tokens"] = longest(state, p), keep
                if L <= LIMIT:
                    break
        p.update({"state": state, "max_input_tokens": L, "max_input_tokens_untruncated": L0, "truncation": info,
                  "returns_truncated": info["returns_dropped_steps"] > 0 or info["previous_steps_dropped"] > 0
                  or info["question_cut_to_tokens"] is not None})
        if L > LIMIT:
            excluded.append(pid)
    for n in neg:
        n["returns_truncated"] = pos[n["pos_id"]]["returns_truncated"]

    # ---- value-visibility rule
    lost = []
    for pid, p in pos.items():
        if pid in excluded or not p["returns_truncated"]:
            continue
        g, k = gold[p["task_id"]], p["k"]
        before = build_state(g["q"], g["steps"], g["obs"], k)
        vals = [v for c in g["steps"][k - 1] for v in leaves(c["arguments"])]
        hidden = [v for v in vals if found(v, before) and found(v, p["state"]) is False]
        if hidden:
            p["values_lost"] = [str(v)[:80] for v in hidden]
            lost.append(pid)

    out_dir.mkdir(parents=True, exist_ok=True)
    drop = set(excluded) | set(lost)
    keep_pos = [p for pid, p in pos.items() if pid not in drop]
    keep_neg = [n for n in neg if n["pos_id"] not in drop]
    with open(out_dir / "excluded_value_lost.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"positive": pos[pid], "negatives": negs_of[pid]}, ensure_ascii=False) + "\n" for pid in lost)
    for name, rows in (("positives.jsonl", keep_pos), ("negatives.jsonl", keep_neg)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    with open(out_dir / "excluded_too_long.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"positive": pos[pid], "negatives": negs_of[pid]}, ensure_ascii=False) + "\n" for pid in excluded)
    print(f"[cfb samples] positives {len(keep_pos)}, negatives {len(keep_neg)}; too long {len(excluded)} states; "
          f"value lost {len(lost)} states")


# ---------------------------------------------------------------- step 4: per-argument inputs
def build_items(gold_path, processed_dir, out_dir):
    """Argument-check / drop-variant inputs of every positive and negative step (state = the positive's state)."""
    schema = {tid: {f["name"]: f for f in r["functions"]} for tid, r in read_gold(gold_path).items()}
    pos = {p["pos_id"]: p for p in read_jsonl(processed_dir / "positives.jsonl")}
    neg = read_jsonl(processed_dir / "negatives.jsonl")
    items, calls = [], []
    todo = [("pos", p, p["pos_id"], None) for p in pos.values()] + [("neg", n, n["neg_id"], n) for n in neg]
    for role, x, cid, n in todo:
        p = pos[x["pos_id"]]
        fs = schema[p["task_id"]]
        desc = {k: v["description"] for k, v in fs.items()}
        step = calls_of(x["action_c1"])
        assert step_action(step, desc) == x["action_c2"], cid
        its = step_items(p["state"], x["action_c1"], fs, cid)
        items += [{"item_id": i, "context": ctx, "question": q} for i, ctx, q in its]
        pids = [i for i, _, _ in its if "|p|" in i]
        dids = [i for i, _, _ in its if "|d|" in i]
        nid = any(k not in ((fs[cl["name"]].get("parameters") or {}).get("properties") or {})
                  for cl in step for k in cl["arguments"])
        calls.append({"call_id": cid, "role": role, "pos_id": p["pos_id"], "neg_id": n["neg_id"] if n else None,
                      "group": n["perturb_type"] if n else None, "query_id": p["query_id"], "k": p["k"],
                      "n_calls": len(step), "n_params": len(pids), "param_items": pids, "drop_items": dids,
                      "param_not_in_def": nid, "returns_truncated": p["returns_truncated"]})
    from deref.build.items import write_items
    write_items(out_dir, items, calls)
    print(f"[cfb items] calls {len(calls)}, inputs {len(items)}")
