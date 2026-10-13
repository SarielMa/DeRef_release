"""Cross-dataset state export: one row per kept state with its question, history, shown window and every scored call.

Cleaning (rows removed with all their calls):
  FC-RewardBench: tasks whose only message is a system prompt (hand-made list of 35 test ids, see
                  `config/fcrb_system_prompt_only_tasks.json`);
  ComplexFuncBench: tasks in the AgentSuite issue list or filter list;
  FinTrace: questions with an unfilled template (TEMPLATE; 6 questions).
Call groups per state (C2 main calls only): FC-RewardBench / ComplexFuncBench "function name" and "parameter value";
FinTrace "P2" (endpoint substitution) and "P4" (argument error). All groups take part in the 2,048-token rule; only
"parameter value" / "P4" form the argument-level analysis sets.
Argument labels: FC-RewardBench wrong call -> wrong iff the BFCL checker rejects the argument; ComplexFuncBench /
FinTrace wrong call -> wrong iff the argument differs from the paired correct call (`wrong_items`); arguments of
correct calls are "correct"; arguments of "function name" / "P2" calls get "function-name group".
"""
import csv
import json
import re
from pathlib import Path

from deref.build import cfb as cfb_mod
from deref.build import fcrb as fcrb_mod
from deref.build import fintrace as ft_mod
from deref.build.items import read_jsonl

CONFIG = Path(__file__).resolve().parent / "config"
SYSTEM_ONLY = CONFIG / "fcrb_system_prompt_only_tasks.json"
TEMPLATE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_ .\-]*\}|\[[A-Z_]{3,}\]|<[A-Z_]{3,}>")
NAMES = {"fcrb": "FC-RewardBench", "cfb": "ComplexFuncBench", "fintrace": "FinTrace"}
GROUPS = {"fcrb": [("function name", "func_name"), ("parameter value", "param_value")],
          "cfb": [("function name", "func_name"), ("parameter value", "param_value")],
          "fintrace": [("P2", "tool"), ("P4", "param")]}


def dumps(o):
    return json.dumps(o, ensure_ascii=False)


def parse_c2(action_c2):
    """[(name, args, description)] from 'call name(args) — description' lines."""
    out = []
    for line in action_c2.split("\n"):
        m = re.match(r"^call (.*?)\((\{.*\})?\) — (.*)$", line, flags=re.S)
        out.append((m.group(1), json.loads(m.group(2)) if m.group(2) else {}, m.group(3)))
    return out


def calls(action_c1):
    """[(name, args)] from 'call name(args)' lines (names may contain "("; empty call = 'call none()')."""
    out = []
    for line in action_c1.split("\n"):
        m = re.match(r"^call (.*?)\((\{.*\})?\)$", line, flags=re.S)
        out.append((m.group(1), json.loads(m.group(2)) if m.group(2) else {}))
    return out


def item_key(item_id):
    """(call id, call index or None, param) of an argument id."""
    cid, k = item_id.split("|p|")
    if ":" in k and k.split(":", 1)[0].isdigit():
        j, k = k.split(":", 1)
        return cid, int(j), k
    return cid, None, k


def wrong_items(pos_action, neg_action, nid, cfb):
    """Arguments of the wrong call that differ from the paired correct call. Calls are paired by name in order (first
    unused correct call of the same name); an argument is wrong if it is missing in the paired call, unequal, or of a
    different type."""
    pc, used, out = calls(pos_action), set(), set()
    for jn, (name, args) in enumerate(calls(neg_action)):
        i = next((i for i, (n, _) in enumerate(pc) if n == name and i not in used), None)
        if i is not None:
            used.add(i)
        pa = pc[i][1] if i is not None else {}
        for k, v in args.items():
            if k not in pa or pa[k] != v or type(pa[k]) is not type(v):
                out.add(f"{nid}|p|{jn}:{k}" if cfb else f"{nid}|p|{k}")
    return out


def load_calls(work, d):
    """(positives by pos_id, negatives in file order with `group` and `c2_main`) of one dataset."""
    pos = {p["pos_id"]: p for p in read_jsonl(work / d / "processed" / "positives.jsonl")}
    neg = read_jsonl(work / d / "processed" / "negatives.jsonl")
    for n in neg:
        if d == "fintrace":
            t = n["perturb_type"]
            n["group"] = "tool" if t.startswith("P2_bin") else ("param" if t == "P4" else t)
            n["c2_main"] = pos[n["pos_id"]]["desc_source"] == "fmp_docs" and n["desc_source"] == "fmp_docs"
        else:
            n["group"] = n["perturb_type"]
            n["c2_main"] = n["action_c2"] is not None
    return pos, neg


def export(work, fcrb_raw, cfb_gold, testset, agentsuite_dir, out_dir, system_only=SYSTEM_ONLY):
    """-> out_dir/export.jsonl (one row per kept state) and out_dir/cleaning.json."""
    with open(system_only, encoding="utf-8") as f:
        fc_drop = set(json.load(f))
    cfb_drop = set()
    for name in ("cfbench_issues.csv", "cfbench_filter_ids.csv"):
        with open(agentsuite_dir / name, encoding="utf-8") as f:
            cfb_drop |= {r["task_id"] for r in csv.DictReader(f)}
    bfcl = {r["item_id"]: r for r in read_jsonl(work / "fcrb" / "bfcl_param_labels.jsonl")}
    recs = {str(x["id"]): x for x in ft_mod.load_testset(testset)}
    ft_tmpl = {i for i, x in recs.items() if TEMPLATE.search(x["source_query"])}
    cfb_gold_steps, cfb_s = {}, {}
    for tid, r in cfb_mod.read_gold(cfb_gold).items():
        cfb_gold_steps[tid] = cfb_mod.gold_steps(r["conversations"])
        cfb_s[tid] = {f["name"]: f for f in r["functions"]}
    fc_s = fcrb_mod.tool_catalogs(fcrb_raw)
    cleaning = []
    out_dir.mkdir(parents=True, exist_ok=True)
    f = open(out_dir / "export.jsonl", "w", encoding="utf-8")
    for d in ("fcrb", "cfb", "fintrace"):
        dname = NAMES[d]
        pos, neg = load_calls(work, d)
        negd = {n["neg_id"]: n for n in neg}
        cl_rows = {c["call_id"]: c for c in read_jsonl(work / d / "items" / "calls.jsonl")}
        items = {r["item_id"]: r for r in read_jsonl(work / d / "items" / "items.jsonl")}
        if d != "cfb":
            items.update({r["item_id"]: r for r in read_jsonl(work / d / "items_nodesc" / "items.jsonl")})

        def reason(pid):
            if d == "fcrb" and pid in fc_drop:
                return "system-prompt-only task"
            if d == "cfb" and pos[pid]["task_id"] in cfb_drop:
                return "AgentSuite issue / filter list"
            if d == "fintrace" and str(pos[pid]["query_id"]) in ft_tmpl:
                return "unfilled template in the question"
            return None

        keep = {}
        for glab, gval in GROUPS[d]:
            negs = [n for n in neg if n["group"] == gval and n["c2_main"]]
            kept = [n for n in negs if reason(n["pos_id"]) is None]
            removed = {}
            for n in negs:
                r = reason(n["pos_id"])
                if r:
                    removed[r] = removed.get(r, 0) + 1
            cleaning.append({"dataset": dname, "group": glab, "positives_before": len({n["pos_id"] for n in negs}),
                             "negatives_before": len(negs), "positives_after": len({n["pos_id"] for n in kept}),
                             "negatives_after": len(kept), "removed_negatives_by_reason": removed})
            for n in kept:
                keep.setdefault(n["pos_id"], []).append((n["neg_id"], glab))
        for pid, negs in keep.items():
            p = pos[pid]
            q = p["state"].split("\nPrevious calls", 1)[0].removeprefix("Question: ")
            if d == "cfb":
                st, ob = cfb_gold_steps[p["task_id"]]
                k = int(p["k"])
                prev = [[(f"- {c['name']}({dumps(c['arguments'])})",
                          dumps(ob[i][j]) if ob[i] and j < len(ob[i]) and ob[i][j] is not None else None)
                         for j, c in enumerate(st[i])] for i in range(max(0, k - 1 - 3), k - 1)]
                header = "Previous calls (most recent 3 steps):"
                hist, names = [("question", q)], [None]  # (label, text), oldest first; function name per piece
                for i in range(k - 1):
                    for j, c in enumerate(st[i]):
                        hist.append((f"call, step {i + 1}", dumps(c["arguments"])))
                        names.append(c["name"])
                        o = ob[i][j] if ob[i] and j < len(ob[i]) else None
                        if o is not None:
                            hist.append((f"return, step {i + 1}", o if isinstance(o, str) else dumps(o)))
                            names.append(None)
            elif d == "fintrace":
                g = ft_mod.golden_calls(recs[str(p["query_id"])])
                pv = [c for c in g[:p["call_idx"]] if c["turn"] < p["turn"]]
                shown = [ln for ln in p["state"].split("\n") if ln.startswith("- ")]
                assert shown == [f"- {c['endpoint']}({dumps(c['args'])})" for c in pv[-5:]], pid
                prev = [[(f"- {c['endpoint']}({dumps(c['args'])})", c["result"])] for c in pv[-5:]]
                header = "Previous calls (most recent 5):"
                hist, names = [("question", q)], [None]
                for c in pv:
                    hist.append((f"call, step {c['turn'] + 1}", dumps(c["args"])))
                    names.append(c["endpoint"])
                    if c["result"] is not None:
                        hist.append((f"return, step {c['turn'] + 1}", c["result"]))
                        names.append(None)
            else:
                prev, header, hist, names = [], "Previous calls (most recent 5):", [("question", q)], [None]
            rows = []
            for cid, role, glab in [(pid, "pos", None)] + [(n, "neg", g) for n, g in negs]:
                x = negd[cid] if role == "neg" else p
                parsed = parse_c2(x["action_c2"])
                schemas = []
                for nm, args, _ in parsed:
                    if d == "fintrace":
                        schemas.append({})
                        continue
                    fdef = (cfb_s[p["task_id"]].get(nm, {}) if d == "cfb" else fc_s[pid].get(nm, {}))
                    props = ((fdef.get("parameters") or {}).get("properties") or {})
                    schemas.append({a: props[a] for a in args if a in props})
                cl = cl_rows[cid]
                if d == "fcrb":
                    wrong = {k for k in cl["param_items"] if bfcl.get(k, {}).get("bfcl_correct") is False} if role == "neg" else set()
                elif role == "neg":
                    wrong = wrong_items(p["action_c1"], x["action_c1"], cid, d == "cfb") & set(cl["param_items"])
                else:
                    wrong = set()
                its = []
                for it in cl["param_items"]:
                    _, jx, prm = item_key(it)
                    v = calls(x["action_c1"])[jx or 0][1].get(prm)
                    lab = ("function-name group" if role == "neg" and glab in ("function name", "P2")
                           else "wrong" if it in wrong else "correct")
                    its.append({"item_id": it, "kind": "param", "call_index": jx or 0, "param": prm, "value": v,
                                "context": items[it]["context"], "question": items[it]["question"], "label": lab})
                for it in cl["drop_items"]:
                    _, jx, prm = item_key(it.replace("|d|", "|p|"))
                    its.append({"item_id": it, "kind": "drop", "call_index": jx or 0, "param": prm,
                                "context": items[it]["context"], "question": items[it]["question"]})
                rows.append({"call_id": cid, "role": role, "group": glab, "action_c1": x["action_c1"],
                             "action_c2": x["action_c2"], "parsed": parsed, "schemas": schemas, "items": its})
            f.write(dumps({"dataset": dname, "pos_id": pid, "question": q, "old_state": p["state"], "header": header,
                           "prev": prev, "history": hist, "history_names": names, "calls": rows}) + "\n")
        print(f"[export] {dname}: states {len(keep)}")
    f.close()
    with open(out_dir / "cleaning.json", "w", encoding="utf-8") as g:
        json.dump(cleaning, g, indent=1)
