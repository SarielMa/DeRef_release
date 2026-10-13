"""State rendering and the per-argument tail of the judge inputs.

Plain judge input of an argument = Plain state + argument tail, question `Q_VALUE`.

Plain state: "Question: <request>\\n<window header>\\n" + the window calls (`- name(args)`), each followed by its return
cut to 1,000 characters (`  -> <return[:1000]>`), or "none". Window: ComplexFuncBench last 3 gold steps, FinTrace last 5
calls, FC-RewardBench none (stored per record in `plain_window`).

2,048-token ladder (CLM input length = Qwen3-8B tokens, special tokens added, of `clm.schema.state_text(context,
question)`): (1) unchanged, (2) returns of the 1, 2, ... earliest shown steps dropped (lines removed), (3) all returns
and the 1, 2, ... earliest steps dropped, (4) the question cut in the middle in steps of 8 tokens down to 40 tokens
(halves joined by " … "). A positive call and its wrong call share one state; the rung of the shared state depends on
every CLM input of the state, including calls outside the data package, so it is stored in the record (`plain_rung`).
Reference-reading inputs walk the ladder from that rung onwards (`further`).

Argument tail: "\\nFunction: <name> — <description>\\nParameter: <param> — <info><schema addition>\\nProposed value for
<param>: <value JSON>".
"""
import json
import re
from pathlib import Path

from clm.schema import state_text

from deref.common import read_jsonl

SYSTEM = "You are a careful evaluator of tool calls made by an AI assistant."
Q_VALUE = "Is this the correct value for this parameter, given the question?"
NOUL_Q = "Is this tool call a correct next step for answering the question?"
PAIR_Q = "Which tool call is a correct next step for answering the question?"
LIMIT, CUT, MIN_Q = 2048, 1000, 40
NO_DESC = "(no description available)"   # parameter missing from the schema (all datasets); CFB: also no description
FCRB_NO_DESC = "(no description)"         # FC-RewardBench parameter in the schema without a description
VALUE_LINE = "\nProposed value for "

# Tokenizers (Hugging Face repo id, pinned revision).
QWEN3_8B = ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")
QWEN3_8_27B = ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")


def load_tokenizer(spec=QWEN3_8B, path=None):
    """AutoTokenizer for (repo id, revision); `path` (a local snapshot directory) overrides the repo id."""
    from transformers import AutoTokenizer
    if path:
        return AutoTokenizer.from_pretrained(str(path))
    return AutoTokenizer.from_pretrained(spec[0], revision=spec[1])


def dumps(o):
    return json.dumps(o, ensure_ascii=False)


def clm_tokens(tok, context, question):
    """CLM input length: Qwen3-8B tokens (special tokens added) of state_text(context, question)."""
    return len(tok(state_text(context, question), add_special_tokens=True)["input_ids"])


# ---------------------------------------------------------------- schema text
def schema_fields(spec, dataset):
    """FC-RewardBench: type / enum / default; ComplexFuncBench: enum only; FinTrace: none (it has no schema here)."""
    if dataset == "cfb":
        return [("enum", dumps(spec["enum"]))] if spec.get("enum") else []
    if dataset == "fintrace":
        return []
    out = []
    if spec.get("type") not in (None, ""):
        out.append(("type", str(spec["type"])))
    if spec.get("enum"):
        out.append(("enum", dumps(spec["enum"])))
    if spec.get("default") not in (None, "", []):
        out.append(("default", dumps(spec["default"])))
    return out


def param_addition(spec, dataset):
    """'; type: …; enum: […]; default: …' appended to the Parameter line ('' if no field exists)."""
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


def properties(rec, name):
    """JSON-schema `properties` of function `name` (FC-RewardBench, ComplexFuncBench); {} if absent."""
    fdef = (rec.get("functions") or {}).get(name) or {}
    return (fdef.get("parameters") or {}).get("properties") or {}


def call_schemas(rec):
    """Per call of the record: {argument: schema} for the arguments present in the schema (FinTrace: {})."""
    out = []
    for c in rec["calls"]:
        if rec["dataset"] == "fintrace":
            out.append({})
            continue
        props = properties(rec, c["name"])
        out.append({a: props[a] for a in c["arguments"] if a in props})
    return out


def param_info(rec, call, param):
    """Text after 'Parameter: <param> — ' (before the schema addition)."""
    d = rec["dataset"]
    if d == "fintrace":
        fdef = (rec.get("functions") or {}).get(call["name"]) or {}
        q = {p["name"]: p for p in fdef.get("params") or []}.get(param)
        if q is None:
            return NO_DESC
        return f"type: {q.get('type')}; required: {str(q.get('required')).lower()}; example: {q.get('example')}"
    props = properties(rec, call["name"])
    if d == "cfb":
        return (props.get(param) or {}).get("description") or NO_DESC
    if param not in props:
        return NO_DESC
    return (props[param] or {}).get("description") or FCRB_NO_DESC


def argument_tail(rec, arg):
    """'\\nFunction: … \\nParameter: … \\nProposed value for …' of one argument (appended to the state)."""
    call = rec["calls"][arg["call_index"]]
    name, param = call["name"], arg["name"]
    add = param_addition(call_schemas(rec)[arg["call_index"]].get(param), rec["dataset"])
    return (f"\nFunction: {name} — {call['description']}\nParameter: {param} — {param_info(rec, call, param)}{add}"
            f"{VALUE_LINE}{param}: {dumps(arg['value'])}")


# ---------------------------------------------------------------- state and ladder
def window(rec):
    """Shared state of a record: {"question", "header", "prev": [[(call line, return or None), ...] per step]}."""
    w = rec["plain_window"]
    return {"question": rec["request"], "header": w["header"],
            "prev": [[(c["call"], c["return"]) for c in step] for step in w["steps"]]}


def build_state(s, drop_ret=0, drop_steps=0, q=None):
    lines = []
    for t, step in enumerate(s["prev"][drop_steps:]):
        for call, ret in step:
            lines.append(call)
            if ret is not None and t >= drop_ret:
                lines.append(f"  -> {ret[:CUT]}")
    return f"Question: {s['question'] if q is None else q}\n{s['header']}\n" + ("\n".join(lines) if lines else "none")


def ladder(s, tok):
    """Truncation steps in order: (label, state text)."""
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


def further(s, tok, lab):
    """The ladder from the shared state's rung `lab` onwards (inputs built on a state never use a longer state)."""
    seen = False
    for step in ladder(s, tok):
        seen = seen or step[0] == lab
        if seen:
            yield step


def plain_state(rec):
    """(rung label, Plain state text) of the record's state at its stored rung."""
    r = rec["plain_rung"]
    return r["label"], build_state(window(rec), r["drop_returns"], r["drop_steps"], r.get("question"))


def select_rung(s, tok, texts_of_state):
    """Rung selection for a new state: first rung at which every CLM input of the state, `texts_of_state(S)` (a list
    of state_text strings), has at most LIMIT tokens. Returns (label, state) or None (state excluded)."""
    ntok = lambda xs: [len(i) for i in tok(xs, add_special_tokens=True)["input_ids"]]  # noqa: E731
    ntok0 = lambda x: len(tok(x, add_special_tokens=False)["input_ids"])  # noqa: E731
    S0 = build_state(s)
    extra = max(ntok(texts_of_state(S0))) - ntok0(S0)
    for lab, S in ladder(s, tok):
        if ntok0(S) + extra > LIMIT + 4:  # cheap pre-check (± boundary tokens)
            continue
        if max(ntok(texts_of_state(S))) <= LIMIT:
            return lab, S
    return None


# ---------------------------------------------------------------- full history
def history(rec):
    """[(label, text)] with the request first: 'question', 'call, step k' (arguments JSON), 'return, step k'."""
    return [("question", rec["request"])] + [(p["label"], p["text"]) for p in rec["history"]]


def pieces(rec):
    """As `history`, with call pieces rendered as 'name(arguments JSON)' (full-history renders, evidence search)."""
    return [("question", rec["request"])] + [(p["label"], f"{p['name']}({p['text']})" if p.get("name") else p["text"])
                                             for p in rec["history"]]


DATASETS = ("fcrb", "cfb", "fintrace")


def load_records(data_dir, dataset):
    """Records of one dataset from the data package (`<data_dir>/data/<dataset>.jsonl` or `<data_dir>/<dataset>.jsonl`)."""
    d = Path(data_dir)
    p = d / "data" / f"{dataset}.jsonl"
    return read_jsonl(p if p.exists() else d / f"{dataset}.jsonl")


def arguments(records, analysis_only=True):
    """[(record, argument)] in record order."""
    return [(r, a) for r in records for a in r["arguments"] if a["in_analysis"] or not analysis_only]
