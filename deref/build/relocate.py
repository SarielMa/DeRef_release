"""FCRB-Relocate: a controlled set built from FC-RewardBench (Figure 2a).

For one correct argument per FC-RewardBench state, the value is moved out of the user request into the return of an added
earlier call:
1. Candidates: correct analysis-set arguments whose value is a string of kind `name`, `identifier` or `contact`
   (`kind`) and is found in the request; one candidate per state, drawn with seed 20260930.
2. Attributes: the value becomes one of the user's saved items. Each item has three attributes of its kind (`ATTRS`);
   one of them is the distinguishing attribute, used by the rewritten request to point at the item.
3. Rewrite: Qwen3.8-27B (thinking off, greedy, 400 new tokens) rewrites the request so that the value is replaced by an
   indirect description that contains the distinguishing attribute value (`PROMPT`).
4. Return: `find_saved_items({"type": <param>})` returns eight entries, the correct one and seven distractors whose
   values are values of other FC-RewardBench samples (same parameter name first, then same function, then any
   parameter). The entries carry `notes` filler so that the correct entry of the buried version starts past
   character 1,000. Two versions differ only in the order of the entries: `visible` (correct entry second, within the
   first 1,000 characters that the judge sees) and `buried` (correct entry last, past the cut). The first distractor
   stays in view in both versions.
5. Candidate values judged per state and version: `pos` (the correct value), `H` (the value of the first distractor,
   in view in both versions) and `C` (a value of the same parameter, or of the same function, from another sample that
   appears nowhere in the input; missing for some states).
States are dropped when the rewrite fails a check, fewer than seven distractors are admissible, the distinguishing
value is not unique to the correct entry, the visible / buried check fails, an input is over 2,048 CLM tokens, or a
distractor equals the correct value after normalisation (lower case, letters and digits only).

All string matching uses `deref.build.paths.found` (the derivation-path matcher)."""
import json
import re
from collections import Counter

import numpy as np

from deref.build import paths as PC
from deref.inputs import state as ST

SEED, MAX_NEW, CUT, LIMIT = 20260930, 400, 1000, 2048
HEADER = "Previous calls (most recent 5):"
FN = "find_saved_items"
FILLER = "No further details recorded."
VERSIONS = ("visible", "buried")
ROLES = ("pos", "H", "C")
DATE = re.compile(r"^\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?$|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$|"
                  r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", re.I)
TIME = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
RANGE = re.compile(r"^\d+[-/]\d+$")
DATES = [f"2024-{m:02d}-28" for m in range(1, 13)]
TAGS = "favorites work weekend holiday gift family travel school party fitness kids archive".split()
CITIES = "Austin Boston Chicago Denver Houston Miami Phoenix Portland Seattle Atlanta Dallas Detroit".split()
TEAMS = "Alpha Bravo Cobalt Delta Ember Falcon Granite Harbor Indigo Juniper Krypton Lumen".split()
OWNERS = ["Alice Moreno", "Ben Okafor", "Chen Wei", "Dana Ross", "Elif Kaya", "Farah Haddad", "Gabe Lund", "Hana Sato",
          "Ivan Petrov", "Jade Brooks", "Kofi Mensah", "Lena Vogel"]
LABELS = "work personal billing support sales legal events press research travel finance admin".split()
ATTRS = {"name": [("saved_on", DATES), ("tag", TAGS), ("city", CITIES)],
         "identifier": [("created_on", DATES), ("team", TEAMS), ("region", CITIES)],
         "contact": [("added_on", DATES), ("owner", OWNERS), ("label", LABELS)]}
PROMPT = """Rewrite a user's request so that one value is no longer written in it. The value is one of the user's saved items,
and the user will now refer to it by one of its attributes.

Value to remove: {value}  (the argument "{param}" of the function "{fn}")
Attribute to use: {d} = {dv}
Original request:
{question}

Instructions:
1. Replace every mention of the value with a short indirect description that contains the attribute value
   "{dv}" written exactly like this (for example "the item I saved under {dv}").
2. The value itself, or any part of it, must not appear in the rewritten request.
3. Keep every other word of the request unchanged. In particular keep these values exactly as they are: {keep}.
Answer with the rewritten request only."""


def nz(s):
    return " ".join(str(s).lower().split())


def nrm(x):
    return re.sub(r"[^0-9a-z]", "", str(x).lower())


def kind(v):
    """'contact', 'identifier', 'name' or None (values that cannot be relocated)."""
    if not isinstance(v, str) or len(v.strip()) < 4:
        return None
    s = v.strip()
    if PC.numeric(s) or s.lower() in ("true", "false", "none", "null") or DATE.search(s) or TIME.match(s) or RANGE.match(s):
        return None
    if "@" in s or "://" in s or s.lower().startswith("www."):
        return "contact"
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/#+\-]*", s) and (re.search(r"\d", s) or re.search(r"[_.:/#+\-]", s) or s.isupper()):
        return "identifier"
    toks = s.split()
    if 1 <= len(toks) <= 6 and all(t[0].isupper() or t[0].isdigit() for t in toks) and any(t[0].isupper() for t in toks):
        return "name"
    return None


# ---------------------------------------------------------------- FC-RewardBench view of the data package
def fcrb_view(records):
    """(positive record by state, correct analysis-set argument ids) of the FC-RewardBench data-package records."""
    pos, correct = {}, set()
    for r in records:
        if r["call_label"] == "correct":
            pos[r["state_id"]] = r
            correct |= {a["item_id"] for a in r["arguments"] if a["in_analysis"] and a["label"] == "correct"}
    return pos, correct


def export_pool(export_rows):
    """(pool, function names) over every FC-RewardBench state of the state export (step 07, all call groups):
    pool = [(state, function, parameter, value)] for every string argument of every call (correct and wrong)."""
    pool, names = [], set()
    for e in export_rows:
        if e["dataset"] != "FC-RewardBench":
            continue
        for c in e["calls"]:
            for nm, _, _ in c["parsed"]:
                names.add(nm)
            for it in c["items"]:
                if it["kind"] == "param" and isinstance(it["value"], str):
                    pool.append((e["pos_id"], c["parsed"][it["call_index"]][0], it["param"], it["value"]))
    return pool, names


def candidates(pos, correct):
    """({state: [candidate]}, [chosen candidate per state, sorted by state])."""
    rows = {}
    for sid, r in pos.items():
        for a in r["arguments"]:
            if a["item_id"] not in correct:
                continue
            k = kind(a["value"])
            if k and PC.found(a["value"], r["request"]):
                c = r["calls"][a["call_index"]]
                keep = [v for p, v in c["arguments"].items() if p != a["name"] and PC.found(v, r["request"])]
                rows.setdefault(sid, []).append({
                    "pos_id": sid, "item_id": a["item_id"], "param": a["name"], "value": a["value"], "kind": k,
                    "fn": c["name"], "question": r["request"], "action_c1": r["action"], "call_index": a["call_index"],
                    "keep": keep})
    rng = np.random.default_rng(SEED)
    chosen = []
    for sid in sorted(rows):
        cs = sorted(rows[sid], key=lambda x: x["item_id"])
        chosen.append(cs[rng.integers(len(cs))] if len(cs) > 1 else cs[0])
    return rows, chosen


def pool_ok(x, r):
    return nz(x) not in nz(r["value"]) and nz(x) not in nz(r["question"])


def attrs(r, rng):
    """Distinguishing attribute and the attributes of the correct entry."""
    names = [a for a, _ in ATTRS[r["kind"]]]
    d = names[rng.integers(len(names))]
    gold = {}
    for a, pool in ATTRS[r["kind"]]:
        ok = [x for x in pool if pool_ok(x, r)]
        gold[a] = ok[rng.integers(len(ok))]
    return d, gold


def distractor_attrs(r, rng, d, gold):
    e = {}
    for a, pool in ATTRS[r["kind"]]:
        ok = [x for x in pool if pool_ok(x, r) and not (a == d and x == gold[d])]
        e[a] = ok[rng.integers(len(ok))]
    return e


# ---------------------------------------------------------------- step 1: rewrite prompts
def rewrite_prompts(records, tok27):
    """Rewrite prompts (Qwen3.8-27B chat text, thinking off), one per state. -> (prompt rows, candidate count by param)."""
    pos, correct = fcrb_view(records)
    rows, chosen = candidates(pos, correct)
    out = []
    for i, r in enumerate(chosen):
        d, gold = attrs(r, np.random.default_rng(SEED + i))
        u = PROMPT.format(value=json.dumps(r["value"], ensure_ascii=False), param=r["param"], fn=r["fn"], d=d,
                          dv=gold[d], question=r["question"],
                          keep=", ".join(json.dumps(v, ensure_ascii=False) for v in r["keep"]) or "none")
        text = tok27.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        out.append({"prompt_id": r["pos_id"], "idx": i, "text": text, "d": d, "gold_attrs": gold,
                    "tokens": len(tok27(text, add_special_tokens=False)["input_ids"]), **r})
    byp = Counter(c["param"] for v in rows.values() for c in v)
    return out, byp.most_common()


# ---------------------------------------------------------------- step 2: rewrite generation (GPU)
def generate_rewrites(prompts, out_path):
    """Greedy Qwen3.8-27B generation of the rewrites (as run: vLLM, bf16, engine seed 0, temperature 0, 400 new tokens,
    max_model_len = longest prompt + 416). Rows {"prompt_id", "output", "finished", "new_tokens"}."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from deref.common import write_jsonl
    from deref.generate.models import load_tokenizer
    from deref.generate.vllm_gen import _llm
    tok = load_tokenizer("qwen3_8_27b")
    ids = [tok(p["text"], add_special_tokens=False)["input_ids"] for p in prompts]
    assert all(len(x) == p["tokens"] for x, p in zip(ids, prompts))
    print(f"[gen] {len(prompts)} prompts, max {max(map(len, ids))} tokens")
    llm = _llm("qwen3_8_27b", max(map(len, ids)) + MAX_NEW + 16)
    res = llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids],
                       SamplingParams(temperature=0.0, max_tokens=MAX_NEW, skip_special_tokens=True))
    write_jsonl(out_path, [{"prompt_id": p["prompt_id"], "output": r.outputs[0].text,
                            "finished": r.outputs[0].finish_reason == "stop", "new_tokens": len(r.outputs[0].token_ids)}
                           for p, r in zip(prompts, res)])
    print(f"[done] {len(prompts)} generations; unfinished {sum(r.outputs[0].finish_reason != 'stop' for r in res)}")


# ---------------------------------------------------------------- step 3: build the states
def compact(o):
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def strip_answer(t):
    t = t.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    return t


def with_value(rec, arg, val):
    """Argument tail of `arg` of the positive record with the value replaced."""
    return ST.argument_tail(rec, {**arg, "value": val})


def build_states(records, export_rows, prompts, generations, tok8, raw_text=None):
    """-> (states, items, drop reasons). `export_rows`: the state export of step 07 (distractor pool); `generations`
    {prompt_id: row}; `raw_text` (optional, bytes of the source files) checks that the added function name does not
    occur in the source data."""
    pos, _ = fcrb_view(records)
    pool, names = export_pool(export_rows)
    assert FN not in names and (raw_text is None or FN.encode() not in raw_text), "function name collides"
    states, items, why = [], [], Counter()
    for p in prompts:
        i, r = p["idx"], p
        rng = np.random.default_rng(SEED + i)
        d, gold_a = attrs(r, rng)
        assert (d, gold_a) == (p["d"], p["gold_attrs"])
        g = generations[p["prompt_id"]]
        rq = strip_answer(g["output"])
        v = r["value"]
        if not g["finished"] or not rq:
            why["unfinished or empty rewrite"] += 1
            continue
        if PC.found(v, rq) or nz(v) in nz(rq):
            why["value still in the rewritten request"] += 1
            continue
        if gold_a[d] not in rq:
            why["distinguishing value not verbatim in the rewritten request"] += 1
            continue
        if any(not PC.found(x, rq) for x in r["keep"]):
            why["another argument value of the request lost"] += 1
            continue

        def adm(x):
            return (kind(x) == r["kind"] and nz(x) != nz(v) and not PC.found(x, r["question"])
                    and not PC.found(x, rq) and nz(x) not in nz(rq))
        t1 = sorted({x for sid, fn, pn, x in pool if sid != r["pos_id"] and pn == r["param"] and adm(x)})
        t2 = sorted({x for sid, fn, pn, x in pool if sid != r["pos_id"] and fn == r["fn"] and pn != r["param"] and adm(x)} - set(t1))
        t3 = sorted({x for sid, fn, pn, x in pool if sid != r["pos_id"] and adm(x)} - set(t1) - set(t2))
        dv = []
        for t in (t1, t2, t3):
            t = [x for x in t if nz(x) not in {nz(y) for y in dv}]
            need = 7 - len(dv)
            if need <= 0:
                break
            if t:
                dv += [t[j] for j in rng.choice(len(t), min(need, len(t)), replace=False)]
        if len(dv) < 7:
            why["fewer than 7 admissible distractor values"] += 1
            continue
        D = [{r["param"]: x, **distractor_attrs(r, rng, d, gold_a), "notes": ""} for x in dv]
        G = {r["param"]: v, **gold_a, "notes": ""}
        n_eq = sum(1 for e in D + [G] for a, _ in ATTRS[r["kind"]] if e[a] == gold_a[d])
        if n_eq != 1 or any(nz(e[d]) in nz(rq) for e in D):
            why["distinguishing value not unique to the correct entry"] += 1
            continue
        k = 0
        while compact({"items": D + [G]}).find(compact(G)[:-1]) < CUT:
            k += 1
            note = " ".join([FILLER] * k)
            for e in D + [G]:
                e["notes"] = note
        vis = compact({"items": [D[0], G] + D[1:]})
        bur = compact({"items": D + [G]})
        if not PC.found(v, vis[:CUT]) or PC.found(v, bur[:CUT]) or not PC.found(v, bur):
            why["visible / buried check failed"] += 1
            continue
        hval = D[0][r["param"]]
        assert PC.found(hval, vis[:CUT]) and PC.found(hval, bur[:CUT])
        call = f"- {FN}(" + json.dumps({"type": r["param"]}, ensure_ascii=False) + ")"
        allt = r["question"] + "\n" + rq + "\n" + call + "\n" + bur
        c1 = [x for x in t1 if x not in dv and not PC.found(x, allt)]
        c2 = [x for x in t2 if x not in dv and not PC.found(x, allt)]
        cval = c1[rng.integers(len(c1))] if c1 else (c2[rng.integers(len(c2))] if c2 else None)
        prec = pos[r["pos_id"]]
        rec = {"pos_id": r["pos_id"], "idx": i, "param": r["param"], "kind": r["kind"], "fn": r["fn"], "value": v,
               "H": hval, "C": cval, "d": d, "gold_attrs": gold_a, "question": r["question"], "rewritten_question": rq,
               "call": call, "call_index": r["call_index"], "padding_k": k, "n_entries": 8, "versions": {}}
        vitems = []
        for ver, ret in (("visible", vis), ("buried", bur)):
            S = f"Question: {rq}\n{HEADER}\n{call}\n  -> {ret[:CUT]}"
            hist = "\n".join([rq, call[2:], ret])
            if cval is not None:
                assert not PC.found(cval, S) and not PC.found(cval, hist)
            p1 = "visible" if PC.found(v, PC.content_text(S, HEADER)) else "history_only" if PC.found(v, hist) else "other"
            rec["versions"][ver] = {"state": S, "return": ret, "path": p1}
            for role, val in (("pos", v), ("H", hval), ("C", cval)):
                if val is None:
                    continue
                for a in prec["arguments"]:
                    rel = a["name"] == r["param"] and a["call_index"] == r["call_index"]
                    tail = with_value(prec, a, val) if rel else ST.argument_tail(prec, a)
                    vitems.append({"item_id": f"{r['pos_id']}|reloc|{ver}|{role}|{a['item_id'].split('|p|', 1)[1]}",
                                   "version": ver, "role": role, "param": a["name"], "relocated": rel,
                                   "label": ("correct" if role == "pos" else "wrong") if rel else "correct",
                                   "context": S + tail, "question": ST.Q_VALUE})
        L = [ST.clm_tokens(tok8, x["context"], x["question"]) for x in vitems]
        for x, n in zip(vitems, L):
            x["tokens"] = n
        if max(L) > LIMIT:
            why["over 2,048 CLM tokens"] += 1
            continue
        ds = [e[r["param"]] for e in json.loads(bur)["items"] if e[r["param"]] != v]
        assert len(ds) == 7
        if any(nrm(x) == nrm(v) for x in ds):
            why["a distractor equals the correct value after normalisation"] += 1
            continue
        states.append(rec)
        items += vitems
    return states, items, why


# ---------------------------------------------------------------- data-package records
def action_with(action, fn, ci, param, val):
    lines = action.split("\n")
    a = json.loads(lines[ci][len("call " + fn) + 1:-1])
    a[param] = val
    lines[ci] = f"call {fn}({json.dumps(a, ensure_ascii=False)})"
    return "\n".join(lines)


def to_records(states, records):
    """One data-package record per state, version and candidate role (FC-RewardBench record layout, dataset
    `fcrb_relocate`). Only the relocated argument is in the analysis set."""
    pos, _ = fcrb_view(records)
    out = []
    for s in states:
        prec = pos[s["pos_id"]]
        for ver in VERSIONS:
            vv = s["versions"][ver]
            for role in ROLES:
                val = s["value"] if role == "pos" else s[role]
                if val is None:
                    continue
                calls = json.loads(json.dumps(prec["calls"], ensure_ascii=False))
                calls[s["call_index"]]["arguments"][s["param"]] = val
                args = []
                for a in prec["arguments"]:
                    rel = a["name"] == s["param"] and a["call_index"] == s["call_index"]
                    args.append({"item_id": f"{s['pos_id']}|reloc|{ver}|{role}|{a['item_id'].split('|p|', 1)[1]}",
                                 "call_index": a["call_index"], "name": a["name"], "value": val if rel else a["value"],
                                 "label": ("correct" if role == "pos" else "wrong") if rel else "correct",
                                 "path": vv["path"] if rel and role == "pos" else None, "relocated": rel,
                                 "in_analysis": rel})
                out.append({
                    "dataset": "fcrb_relocate", "item_id": f"{s['pos_id']}|reloc|{ver}|{role}", "query_id": s["pos_id"],
                    "state_id": f"{s['pos_id']}|reloc|{ver}", "source_state_id": s["pos_id"], "version": ver,
                    "role": role, "call_label": "correct" if role == "pos" else "wrong",
                    "request": s["rewritten_question"], "original_request": s["question"],
                    "history": [{"label": "call, step 1", "kind": "call", "step": 1, "name": FN,
                                 "text": json.dumps({"type": s["param"]}, ensure_ascii=False)},
                                {"label": "return, step 1", "kind": "return", "step": 1, "name": None, "text": vv["return"]}],
                    "plain_window": {"header": HEADER, "steps": [[{"call": s["call"], "return": vv["return"]}]]},
                    "plain_rung": {"label": "none", "drop_returns": 0, "drop_steps": 0, "question": None},
                    "action": action_with(prec["action"], s["fn"], s["call_index"], s["param"], val),
                    "calls": calls, "functions": prec["functions"], "arguments": args,
                    "relocation": {"param": s["param"], "call_index": s["call_index"], "value_kind": s["kind"],
                                   "correct_value": s["value"], "H": s["H"], "C": s["C"], "attribute": s["d"],
                                   "correct_entry_attributes": s["gold_attrs"], "padding_k": s["padding_k"],
                                   "n_entries": s["n_entries"]}})
    return out
