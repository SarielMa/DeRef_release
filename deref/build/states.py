"""Final state of every kept state and the 2,048-token rule (CLM input limit).

State = "Question: <q>\\n<header>\\n" + the shown window ("- call" lines, each followed by "  -> <return[:1000]>"),
or "none". Schema text (FC-RewardBench: type / enum / default; ComplexFuncBench: enum; FinTrace: none) is added to
argument checks ("; type: …" on the parameter line) and to C2 action texts (" Parameter schema — p: …").

Length rule: every CLM input of the state — native state_text(S), pairwise state_text(S, PAIR_Q), pointwise C1 / C2
of every call (the correct call and ALL its wrong calls, of every group), and every argument check and drop variant
of every call — must be <= 2048 Qwen3-8B tokens (special tokens as added by the tokenizer). Ladder, in order: drop
the returns of the earliest shown step (one by one), then drop the earliest shown steps, then cut the question in the
middle (" … ", steps of 8 tokens, at least 40 kept). The first rung that fits is the state's rung; if none fits, the
state is excluded. (In the released data no state is excluded and no question is cut.)
"""
import json
import re

from deref.build.items import NOUL_Q, PAIR_Q, noul_context

LIMIT, CUT, MIN_Q = 2048, 1000, 40
SHORT = {"FC-RewardBench": "fcrb", "ComplexFuncBench": "cfb", "FinTrace": "fintrace"}
QWEN3_8B = "Qwen/Qwen3-8B"
QWEN3_8B_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def dumps(o):
    return json.dumps(o, ensure_ascii=False)


def clm_text(context, question):
    """Text the CLM embeds for a (context, question) pair: context first, question last (= clm.schema.state_text
    for string inputs)."""
    s, i = (context or "").strip(), (question or "").strip()
    return f"{s}\n\n{i}" if s and i else (s or i)


# ---------------------------------------------------------------- schema text
def schema_fields(spec, dataset):
    """FC-RewardBench: type / enum / default; ComplexFuncBench: enum only; FinTrace: none (no specs exported)."""
    if dataset == "ComplexFuncBench":
        return [("enum", dumps(spec["enum"]))] if spec.get("enum") else []
    out = []
    if spec.get("type") not in (None, ""):
        out.append(("type", str(spec["type"])))
    if spec.get("enum"):
        out.append(("enum", dumps(spec["enum"])))
    if spec.get("default") not in (None, "", []):
        out.append(("default", dumps(spec["default"])))
    return out


def param_addition(spec, dataset):
    f = schema_fields(spec or {}, dataset)
    return ("; " + "; ".join(f"{k}: {v}" for k, v in f)) if f else ""


def action_with_schema(action, schemas, dataset):
    """Append ' Parameter schema — …' to each 'call name(args) — description' line (args as written in the line)."""
    lines = []
    for j, line in enumerate(action.split("\n")):
        m = re.match(r"^call (.*?)\((\{.*\})?\) — ", line, flags=re.S)
        args = json.loads(m.group(2)) if m and m.group(2) else {}
        sc = schemas[j] if j < len(schemas) else {}
        parts = []
        for a in args:
            f = schema_fields(sc.get(a) or {}, dataset)
            if f:
                parts.append(f"{a}: " + ", ".join(f"{k}: {v}" for k, v in f))
        lines.append(line + (" Parameter schema — " + "; ".join(parts) if parts else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------- state and ladder
def build_state(s, drop_ret=0, drop_steps=0, q=None):
    """s: export row (question, header, prev = [[(call line, return or None), ...], ...] oldest step first)."""
    lines = []
    for t, step in enumerate(s["prev"][drop_steps:]):
        for call, ret in step:
            lines.append(call)
            if ret is not None and t >= drop_ret:
                lines.append(f"  -> {ret[:CUT]}")
    return f"Question: {s['question'] if q is None else q}\n{s['header']}\n" + ("\n".join(lines) if lines else "none")


def ladder(s, tok):
    """Truncation rungs in order: (label, state text)."""
    n = len(s["prev"])
    yield ("none", build_state(s))
    for dr in range(1, n + 1):
        yield (f"returns dropped for {dr} earliest step(s)", build_state(s, drop_ret=dr))
    for dsp in range(1, n + 1):
        yield (f"all returns and {dsp} earliest step(s) dropped", build_state(s, drop_ret=n, drop_steps=dsp))
    ids = tok(s["question"], add_special_tokens=False)["input_ids"]
    for keep in range(len(ids) - 8, MIN_Q - 1, -8):
        q2 = tok.decode(ids[:keep // 2]) + " … " + tok.decode(ids[len(ids) - (keep - keep // 2):])
        yield (f"question cut to {keep} tokens", build_state(s, drop_ret=n, drop_steps=n, q=q2))


def inputs(s, S):
    """All CLM texts of export row s under state S, and the per-item final contexts (call, item, text after state)."""
    d, old = s["dataset"], s["old_state"]
    texts, main_items = [clm_text(S, None), clm_text(S, PAIR_Q)], []
    for c in s["calls"]:
        c2 = action_with_schema(c["action_c2"], c["schemas"], d)
        texts += [clm_text(noul_context(S, c["action_c1"]), NOUL_Q), clm_text(noul_context(S, c2), NOUL_Q)]
        for it in c["items"]:
            assert it["context"].startswith(old), it["item_id"]
            rest = it["context"][len(old):]
            if it["kind"] == "param":
                add = param_addition(c["schemas"][it["call_index"]].get(it["param"]) if c["schemas"] else None, d)
                head, tail = rest.rsplit("\nProposed value for ", 1)
                rest = head + add + "\nProposed value for " + tail
            else:
                act = rest.removeprefix("\nProposed call: ")
                rest = "\nProposed call: " + action_with_schema(act, c["schemas"], d)
            main_items.append((c, it, rest))
            texts.append(clm_text(S + rest, it["question"]))
    return texts, main_items


def choose_rung(s, tok):
    """(label, state, per-item contexts, longest input tokens) of the first rung whose inputs all fit, or None."""
    ntok = lambda xs: [len(i) for i in tok(xs, add_special_tokens=True)["input_ids"]]  # noqa: E731
    ntok0 = lambda x: len(tok(x, add_special_tokens=False)["input_ids"])  # noqa: E731
    S0 = build_state(s)
    t0, _ = inputs(s, S0)
    extra = max(ntok(t0)) - ntok0(S0)
    for lab, S in ladder(s, tok):
        if ntok0(S) + extra > LIMIT + 4:  # cheap pre-check (+- boundary tokens)
            continue
        texts, main_items = inputs(s, S)
        L = ntok(texts)
        if max(L) <= LIMIT:
            return lab, S, main_items, max(L)
    return None


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN3_8B, revision=QWEN3_8B_REVISION)


def build_states(export_path, out_dir, tok=None, datasets=None):
    """-> out_dir/<d>/{positives,negatives,calls,excluded}.jsonl (+ labels.jsonl: one row per argument item)."""
    tok = tok or load_tokenizer()
    names = {v: k for k, v in SHORT.items()}
    todo = [names[d] for d in datasets] if datasets else list(SHORT)
    for d in todo:
        (out_dir / SHORT[d]).mkdir(parents=True, exist_ok=True)
    out = {d: {k: open(out_dir / SHORT[d] / f"{k}.jsonl", "w", encoding="utf-8")
               for k in ("positives", "negatives", "calls", "excluded", "labels")} for d in todo}
    n = {}
    with open(export_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            d = s["dataset"]
            if d not in out:
                continue
            chosen = choose_rung(s, tok)
            if chosen is None:
                out[d]["excluded"].write(dumps({"pos_id": s["pos_id"], "reason": "main version still over 2048"}) + "\n")
                continue
            lab, S, _, _ = chosen
            n[d] = n.get(d, 0) + 1
            pos = next(c for c in s["calls"] if c["role"] == "pos")
            out[d]["positives"].write(dumps({"pos_id": s["pos_id"], "query_id": s["pos_id"], "state": S, "truncation": lab,
                                             "action_c1": pos["action_c1"],
                                             "action_c2": action_with_schema(pos["action_c2"], pos["schemas"], d)}) + "\n")
            for c in s["calls"]:
                if c["role"] == "neg":
                    out[d]["negatives"].write(dumps({"neg_id": c["call_id"], "pos_id": s["pos_id"], "group": c["group"],
                                                     "action_c1": c["action_c1"],
                                                     "action_c2": action_with_schema(c["action_c2"], c["schemas"], d)}) + "\n")
                out[d]["calls"].write(dumps({"call_id": c["call_id"], "role": c["role"], "pos_id": s["pos_id"], "group": c["group"],
                                             "param_items": [i["item_id"] for i in c["items"] if i["kind"] == "param"],
                                             "drop_items": [i["item_id"] for i in c["items"] if i["kind"] == "drop"]}) + "\n")
                for it in c["items"]:
                    if it["kind"] == "param":
                        out[d]["labels"].write(json.dumps({"dataset": d, "item_id": it["item_id"], "call_id": c["call_id"],
                                                           "role": c["role"], "label": it["label"]}) + "\n")
    for d in out:
        for fh in out[d].values():
            fh.close()
    print(f"[states] kept {n}")
