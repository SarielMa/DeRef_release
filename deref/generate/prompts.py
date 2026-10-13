"""Generation prompts of every method, built from data-package records.

Methods (thinking-off "original" -> thinking-on version used in the paper):

| prompt file (thinking off) | prompt file (thinking on) | content |
|---|---|---|
| deref_nothink            | deref            | derivation (F2) on the full-history locator context, one per locator slot |
| deref_qwen3_8b_nothink   | deref_qwen3_8b   | the derivation prompt rendered with the Qwen3-8B chat template |
| evidence_search_nothink  | evidence_search  | locator prompt with the value line inserted |
| sgv_prior_nothink        | sgv_prior        | F2 on the Plain (truncated) judge context without the value line |
| revise_nothink           | revise           | revise the proposed call (one per evaluated call) |

A locator slot is (state, Function line, Parameter line, parameter name); its chat text is fitted to 32,768 Qwen3.8-27B
tokens with the deletion ladder of `render.fit`. Derivation prompts edit that fitted text
(they are not re-fitted). Revise prompts are fitted with their own tail. The thinking-on prompts re-render the
same system / user messages and are not re-fitted (they are 3 tokens shorter).

Generation as run for the paper (scripts/22_generate.py): thinking-on files with the thinking mode (8,192 new tokens;
deref_qwen3_8b with Qwen3-8B); thinking-off originals greedily with 1,024 new tokens (evidence_search_nothink: 160;
deref_qwen3_8b_nothink with Qwen3-8B). The ComplexFuncBench sgv_prior_nothink outputs were first generated greedily with
200 new tokens (max_model_len 32,768) and only the unfinished ones were rerun with 1,024 (a single 1,024-token greedy run is
equivalent up to greedy prefix stability). The thinking-on prompts depend only on the original prompt
texts, not on their generations.

Prompt rows: {"prompt_id", "text", "tokens", "dataset"}; map rows: {"item_id", "prompt_id"} for every argument the
method covers. Prompt ids are `<method>|<dataset>|<running number>`; a thinking-on prompt carries the running number
of its thinking-off original.
"""
import json
from collections import OrderedDict

from deref.generate import render as R
from deref.inputs import plain as PL
from deref.inputs import state as ST

NOTHINK = ("deref_nothink", "deref_qwen3_8b_nothink", "evidence_search_nothink", "sgv_prior_nothink", "revise_nothink")
THINK = {m: m[:-len("_nothink")] for m in NOTHINK}
METHODS = NOTHINK + tuple(THINK.values())
LOCATOR = "locator"  # base prompt of the per-argument methods (fitted locator text); not itself a paper method
CTX_8B, MAX_NEW_GREEDY, MAX_NEW_THINKING = 40960, 1024, 8192


def function_definition(dataset, schema, name):
    """Function line and [(param, Parameter line)] of the full function schema, in schema order (Revise)."""
    if dataset == "fintrace":
        ps = [(q["name"], f"type: {q.get('type')}; required: {str(q.get('required')).lower()}; example: {q.get('example')}",
               None) for q in schema.get("params") or []]
        desc = schema["description"]
    else:
        props = (schema.get("parameters") or {}).get("properties") or {}
        nd = ST.FCRB_NO_DESC if dataset == "fcrb" else ST.NO_DESC
        ps = [(k, (v or {}).get("description") or nd, v) for k, v in props.items()]
        desc = schema["description"]
    return f"Function: {name} — {desc}", [(k, f"Parameter: {k} — {i}{ST.param_addition(s, dataset)}") for k, i, s in ps]


class _Table:
    """Deduplicating prompt table of one method and dataset: key -> running number, plus the item map."""

    def __init__(self, method, dataset):
        self.method, self.dataset = method, dataset
        self.rows = OrderedDict()  # key -> row
        self.map = []

    def pid(self, n):
        return f"{self.method}|{self.dataset}|{n:05d}"

    def add(self, item_id, key, make):
        """make() -> (text, tokens) or None (prompt not generated: no map row)."""
        if key not in self.rows:
            r = make()
            self.rows[key] = None if r is None else {"prompt_id": self.pid(len(self.rows)), "text": r[0], "tokens": r[1],
                                                      "dataset": self.dataset}
        row = self.rows[key]
        if row is not None:
            self.map.append({"item_id": item_id, "prompt_id": row["prompt_id"]})
        return row

    def prompts(self):
        return [r for r in self.rows.values() if r is not None]


def build(records, dataset, tok27, tok8=None, methods=METHODS, analysis_only=True, log=print):
    """Prompts and maps of `methods` for one dataset. -> {method: (prompt rows, map rows)}.

    `records` are the data-package records of the dataset; arguments are taken in item_id order (only the analysis
    set unless analysis_only=False). Thinking-on methods need their thinking-off original, which is built as well."""
    want = set(methods)
    for m in list(want):
        if m in THINK.values():
            want.add(m + "_nothink")
    if want & {"deref_qwen3_8b_nothink", "deref_qwen3_8b"}:
        assert tok8 is not None, "Qwen3-8B tokenizer needed"
    T = {m: _Table(m, dataset) for m in METHODS + (LOCATOR,) if m in want}
    n27 = lambda t: R.ntok(tok27, t)  # noqa: E731

    def on27(text):
        """Qwen3.8-27B thinking-on re-render (not re-fitted; asserted within the limit)."""
        on = R.thinking_on(tok27, text)
        n = n27(on)
        assert n <= R.LOC_LIMIT
        return on, n

    states, items = {}, []
    for rec in records:
        if rec["state_id"] not in states:
            states[rec["state_id"]] = {"question": rec["request"], "pieces": ST.pieces(rec)[1:]}
        for a in rec["arguments"]:
            if a["in_analysis"] or not analysis_only:
                items.append((a["item_id"], rec, a))
    items.sort(key=lambda x: x[0])
    log(f"[{dataset}] {len(items)} arguments, {len(states)} states")

    locator = {}  # slot key -> (fitted locator chat text, tokens)

    def loc_fit(key):
        if key not in locator:
            sid, fl, pl, name = key
            _, _, text, n = R.fit(states[sid], R.locator_tail(fl, pl, name), tok27)
            locator[key] = (text, n)
        return locator[key]

    def loc(key):
        return loc_fit(key)[0]

    for done, (k, rec, a) in enumerate(items):
        if done and done % 500 == 0:
            log(f"[{dataset}] {done} / {len(items)} arguments")
        sid = rec["state_id"]
        ctx = PL.plain_item(rec, a)["context"]
        _, fl, pl = R.split_main(ctx)
        name = a["name"]
        slot = (sid, fl, pl, name)
        need_deref = want & {"deref_nothink", "deref", "deref_qwen3_8b_nothink", "deref_qwen3_8b"}
        dtext = R.derivation_text(loc(slot)) if need_deref else None
        if LOCATOR in T:
            T[LOCATOR].add(k, slot, lambda: loc_fit(slot))
        # ---- derivation (thinking off / on)
        if "deref_nothink" in T:
            T["deref_nothink"].add(k, slot, lambda: (dtext, n27(dtext)))
        if "deref" in T:
            T["deref"].add(k, slot, lambda: on27(dtext))
        # ---- Qwen3-8B derivation
        if "deref_qwen3_8b_nothink" in T:
            def c_off():
                sysm, user = R.split_chat(dtext)
                tc = R.chat(tok8, user, sysm)
                assert R.split_chat(tc) == (sysm, user)
                nc = R.ntok(tok8, tc)
                return None if nc + MAX_NEW_GREEDY > CTX_8B else (tc, nc)
            row = T["deref_qwen3_8b_nothink"].add(k, slot, c_off)
            if "deref_qwen3_8b" in T:
                def c_on():
                    sysm, user = R.split_chat(dtext)
                    on = R.thinking_on(tok8, R.chat(tok8, user, sysm), small_model=True)
                    n = R.ntok(tok8, on)
                    if n + MAX_NEW_THINKING > CTX_8B:
                        return None
                    assert n <= R.LOC_LIMIT
                    return on, n
                T["deref_qwen3_8b"].add(k, slot, c_on)
        # ---- evidence search
        if "evidence_search_nothink" in T:
            tl = R.insert_value_line(loc(slot), pl, name, a["value"])
            key = (slot, json.dumps(a["value"], sort_keys=True)) if dataset == "cfb" else tl

            def lv():
                n = n27(tl)
                assert n <= R.LOC_LIMIT
                return tl, n
            T["evidence_search_nothink"].add(k, key, lv)
            if "evidence_search" in T:
                T["evidence_search"].add(k, key, lambda: on27(tl))
        # ---- SGV-style prior
        if "sgv_prior_nothink" in T:
            nm = R.pname(k)
            assert ctx.rsplit(ST.VALUE_LINE, 1)[1].startswith(f"{nm}: "), k
            nv = R.without_value(ctx)
            ta = R.chat(tok27, f"{nv}\n\n{R.F2.format(name=nm)}")
            T["sgv_prior_nothink"].add(k, nv, lambda: (ta, n27(ta)))
            if "sgv_prior" in T:
                T["sgv_prior"].add(k, nv, lambda: on27(ta))
        # ---- Revise
        call = rec["calls"][a["call_index"]]
        fname = call["name"]
        if "revise_nothink" in T:
            def rev():
                fl2, pls = function_definition(dataset, rec["functions"][fname], fname)
                tail = (fl2 + "".join("\n" + p for _, p in pls) + "\nProposed call: " + json.dumps(call["arguments"], ensure_ascii=False)
                        + "\n" + R.RUBRIC + "\n\n" + R.REV)
                _, _, text, n = R.fit(states[sid], tail, tok27)
                return text, n
            rkey = (rec["item_id"], a["call_index"])
            row = T["revise_nothink"].add(k, rkey, rev)
            if "revise" in T:
                T["revise"].add(k, rkey, lambda: on27(row["text"]))
    return {m: (T[m].prompts(), T[m].map) for m in T if m in set(methods)}
