"""FC-RewardBench: source rows -> correct / wrong call pairs -> per-argument inputs -> BFCL argument labels.

Source: HF dataset `ibm-research/fc-reward-bench` at revision FCRB_REVISION, read with `datasets.load_from_disk`
from a `save_to_disk` copy (columns tools, conversation, chosen_output, rejected_output, error_type, model_name,
test_category, test_id). One positive (chosen output) and one negative (rejected output) per row; no sampling.
"""
import collections
import copy
import json
import re
import sys
from pathlib import Path

from deref.build.items import NOT_PARAM, call_items, read_jsonl, reword_not_param, split_call, write_items, write_jsonl

FCRB_REVISION = "269929c3329e603e87ed3203de42896cc03ddbf3"
BFCL_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"   # gorilla-llm/Berkeley-Function-Calling-Leaderboard (HF)
BFCL_CHECKER_COMMIT = "70b6a4a21445"                       # github.com/ShishirPatil/gorilla
STATE = "Question: {q}\nPrevious calls (most recent 5):\nnone"  # state template for a first call
EMPTY_ACTION = "call none()"


def load_raw(raw_dir):
    from datasets import load_from_disk
    return load_from_disk(str(raw_dir))


def tool_catalogs(raw_dir):
    """{test_id: {tool name: tool dict}} from the source rows' `tools` column."""
    return {r["test_id"]: {t["name"]: t for t in json.loads(r["tools"])} for r in load_raw(raw_dir)}


# ---------------------------------------------------------------- correct / wrong call pairs
def group(et):
    """Error group of a row from its BFCL error_type; None = excluded (output parsing error)."""
    if et == "simple_function_checker:wrong_func_name":
        return "func_name"
    if et.startswith("value_error:"):
        return "param_value"
    if et == "parallel_function_checker_no_order:cannot_find_match":
        return "parallel_mismatch"
    if et.endswith(":wrong_count"):
        return "call_count"
    if et in ("simple_function_checker:missing_required", "simple_function_checker:unexpected_param") or et.startswith("type_error:"):
        return "other_param"
    if et == "simple_function_checker:missing_optional":
        return "missing_optional"
    if et == "ast_decoder:decoder_wrong_output_format":
        return None
    raise ValueError(et)


def model_size(name):
    """Parameter count (billions) parsed from the model name, or None when the name has none."""
    m = re.findall(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)[bB](?![A-Za-z])", name.replace("_", "-"))
    return float(m[-1]) if m else None


def size_group(name):
    s = model_size(name)
    return "unknown" if s is None else ("<=1B" if s <= 1 else ">1B")


def split_calls(raw):
    """[(name, raw args text)] taken from the raw output string. Every call's raw text is exactly
    json.dumps({name: args}) (asserted), so the args substring is json.dumps(args) with the stored escapes."""
    v = json.loads(raw)
    out = [(next(iter(x)), json.dumps(next(iter(x.values())))) for x in v]
    rebuilt = "[" + ", ".join(json.dumps({n: json.loads(a)}) for n, a in out) + "]"
    assert rebuilt == raw, raw[:200]
    for n, a in out:  # the args text must literally occur in the raw string
        assert f'{json.dumps(n)}: {a}' in raw
    return out


def actions(calls, catalog):
    """(C1 text, C2 text or None). C2 is None if any called function is missing from the row's catalog."""
    if not calls:
        return EMPTY_ACTION, EMPTY_ACTION
    c1 = "\n".join(f"call {n}({a})" for n, a in calls)
    if not all(n in catalog for n, _ in calls):
        return c1, None
    c2 = "\n".join(f"call {n}({a}) — {catalog[n]['description']}" for n, a in calls)
    return c1, c2


def build_samples(raw_dir, out_dir):
    """-> out_dir/positives.jsonl, negatives.jsonl (one pair per kept row)."""
    ds = load_raw(raw_dir)
    pos, neg, counts = [], [], collections.Counter()
    for r in ds:
        g = group(r["error_type"])
        if g is None:
            counts["excluded: parsing error"] += 1
            continue
        if r["chosen_output"] == r["rejected_output"]:
            counts["excluded: identical raw outputs"] += 1
            continue
        catalog = {t["name"]: t for t in json.loads(r["tools"])}
        ch, rj = split_calls(r["chosen_output"]), split_calls(r["rejected_output"])
        p1, p2 = actions(ch, catalog)
        n1, n2 = actions(rj, catalog)
        assert p2 is not None, f"chosen function missing from catalog: {r['test_id']}"
        assert n1 != p1 and (n2 is None or n2 != p2)
        qid = r["test_id"]
        meta = {"query_id": qid, "test_category": r["test_category"], "error_type": r["error_type"], "group": g,
                "model_name": r["model_name"], "model_size_b": model_size(r["model_name"]),
                "size_group": size_group(r["model_name"]), "n_calls_chosen": len(ch), "n_calls_rejected": len(rj),
                "parallel": max(len(ch), len(rj)) > 1, "catalog_size": len(catalog),
                "same_names": collections.Counter(n for n, _ in ch) == collections.Counter(n for n, _ in rj)}
        state = STATE.format(q=r["conversation"][0]["content"])
        pos.append({"pos_id": qid, **meta, "state": state, "endpoints": [n for n, _ in ch], "action_c1": p1, "action_c2": p2})
        neg.append({"neg_id": f"{qid}-neg", "pos_id": qid, "perturb_type": g, **meta, "endpoints": [n for n, _ in rj],
                    "action_c1": n1, "action_c2": n2})
    write_jsonl(out_dir / "positives.jsonl", pos)
    write_jsonl(out_dir / "negatives.jsonl", neg)
    print(f"[fcrb samples] positives {len(pos)}, negatives {len(neg)}, {dict(counts)}")


# ---------------------------------------------------------------- per-argument inputs
def build_items(raw_dir, processed_dir, out_dir):
    """Argument-check / drop-variant inputs for the calls of the function-name and parameter-value groups (C2 only)."""
    raw = {r["test_id"]: r for r in load_raw(raw_dir)}
    pos = {p["pos_id"]: p for p in read_jsonl(processed_dir / "positives.jsonl")}
    neg = [n for n in read_jsonl(processed_dir / "negatives.jsonl")
           if n["perturb_type"] in ("func_name", "param_value") and n["action_c2"] is not None]
    items, calls = [], []
    done = set()
    for n in neg:
        p = pos[n["pos_id"]]
        cat = {t["name"]: t for t in json.loads(raw[n["pos_id"]]["tools"])}
        for role, x in (("pos", p), ("neg", n)):
            cid = p["pos_id"] if role == "pos" else n["neg_id"]
            if cid in done:
                continue
            done.add(cid)
            name, rawa, args = split_call(x["action_c1"])
            assert json.dumps(args) == rawa, cid
            tool = cat[name]
            fdesc = tool["description"]
            assert x["action_c2"] == f"call {name}({rawa}) — {fdesc}", cid
            props = (tool.get("parameters") or {}).get("properties") or {}

            def info_of(k, props=props):
                if k not in props:
                    return NOT_PARAM
                return (props[k] or {}).get("description") or "(no description)"
            its, pids, dids, nid = call_items(cid, p["state"], name, fdesc, args, info_of, rawa)
            items += its
            calls.append({"call_id": cid, "role": role, "pos_id": p["pos_id"], "neg_id": n["neg_id"] if role == "neg" else None,
                          "group": n["perturb_type"] if role == "neg" else None, "query_id": p["query_id"],
                          "n_params": len(args), "param_items": pids, "drop_items": dids, "param_not_in_def": nid})
    write_items(out_dir, items, calls)
    print(f"[fcrb items] calls {len(calls)}, inputs {len(items)}")


def build_items_nodesc(items_dir, out_dir):
    n = reword_not_param(items_dir / "items.jsonl", out_dir / "items.jsonl")
    print(f"[fcrb items, re-worded] {n}")


# ---------------------------------------------------------------- BFCL argument labels
def checker_root(path):
    """Directory holding the `bfcl` package: a gorilla checkout (berkeley-function-call-leaderboard/) or that dir."""
    p = Path(path)
    sub = p / "berkeley-function-call-leaderboard"
    return sub if (sub / "bfcl").is_dir() else p


def calls_of(action_c1):
    out = []
    for line in action_c1.split("\n"):
        m = re.match(r"^call (.*?)\((\{.*\})?\)$", line, flags=re.S)
        out.append((m.group(1), json.loads(m.group(2)) if m.group(2) else {}))
    return out


def bfcl_labels(bfcl_dir, checker, processed_dir, items_dir, out_path):
    """Per-argument label with the official BFCL v3 AST checker.

    Per argument p (value v) of a call to function f, `simple_function_checker` is called with
      func_description = BFCL's description of f with `required` emptied,
      model_output     = {f: {p: v}}   (f in the underscore form FC-RewardBench uses),
      possible_answer  = {f: {p: possible[p]}} if p is in the gold call, else {f: {}},
    so the checker runs exactly its per-parameter block for p. Wrong <=> valid is False. Several gold entries for the
    same function (parallel categories): correct if the argument passes against any of them.
    bfcl_dir holds questions/BFCL_v3_*.json and possible_answer/BFCL_v3_*.json (JSON lines)."""
    sys.path.insert(0, str(checker_root(checker)))
    from bfcl.eval_checker.ast_eval import ast_checker as A
    from bfcl.model_handler.constant import UNDERSCORE_TO_DOT
    model = UNDERSCORE_TO_DOT[0]  # a model whose outputs use "_" for "." (as FC-RewardBench's calls do)
    q, pa = {}, {}
    for f in sorted((bfcl_dir / "questions").glob("*.json")):
        for r in read_jsonl(f):
            q[r["id"]] = r
    for f in sorted((bfcl_dir / "possible_answer").glob("*.json")):
        for r in read_jsonl(f):
            pa[r["id"]] = r["ground_truth"]
    pos = {p["pos_id"]: p for p in read_jsonl(processed_dir / "positives.jsonl")}
    neg = {n["neg_id"]: n for n in read_jsonl(processed_dir / "negatives.jsonl")}
    calls = read_jsonl(items_dir / "calls.jsonl")
    labels, unmatched = [], collections.Counter()
    for cl in calls:
        cid, pid = cl["call_id"], cl["pos_id"]
        if pid not in q or pid not in pa:
            unmatched[pid] += 1
            continue
        fdesc = {A.convert_func_name(f["name"], model): f for f in q[pid]["function"]}
        gold = [(A.convert_func_name(next(iter(g)), model), next(iter(g.values()))) for g in pa[pid]]
        cs = calls_of((neg[cid] if cl["role"] == "neg" else pos[pid])["action_c1"])
        for item in cl["param_items"]:
            p = item.split("|p|", 1)[1]
            name, args = cs[0]
            if name not in fdesc:
                labels.append({"item_id": item, "call_id": cid, "role": cl["role"], "status": "function not in BFCL question"})
                continue
            entries = [g for n, g in gold if n == name]
            d = copy.deepcopy(fdesc[name])
            d["parameters"]["required"] = []
            d["name"] = fdesc[name]["name"]
            ok, errs = False, []
            for g in entries or [{}]:
                r = A.simple_function_checker(d, {name: {p: args[p]}}, {name: ({p: g[p]} if p in g else {})}, "Python", model)
                errs.append(r.get("error_type"))
                ok |= bool(r["valid"])
            labels.append({"item_id": item, "call_id": cid, "role": cl["role"], "status": "matched",
                           "bfcl_correct": ok, "error_types": errs, "n_gold_entries": len(entries),
                           "value": args[p], "possible": [g.get(p) for g in entries]})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"[fcrb bfcl] calls {len(calls)}; positives not in BFCL v3: {len(unmatched)} {sorted(unmatched)}; "
          f"argument labels {len(labels)}")
