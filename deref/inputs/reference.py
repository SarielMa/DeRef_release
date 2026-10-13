"""Reference-reading judge inputs: a generation output turned into a block inserted between the state and the argument
tail of the Plain input.

Block construction per method:
  deref, sgv_prior, deref_qwen3_8b: HEAD + answer (text after the last </think>);
  deref_nothink (thinking off): HEAD + the whole output;
  revise: HEAD_R + the answer's JSON object re-serialised (`parse_block`; raw answer if no object);
  expected_value: HEAD + "Expected value for <name>: <value>." when the expected-value parser reads a definite value from the
      DeRef answer, else "The context does not determine the value of <name>.";
  evidence_search: "\\nSource snippets:\\n" + snippets of the verified quotes of the answer (`evidence.evidence_snippets`).
The state walks the 2,048-token ladder from the Plain rung (`state.further`) until the CLM input fits. An argument
without an answer, without a snippet, or over the limit at every rung gets no item (its Plain score is kept).
Item id = "<argument item_id>|<tag>"."""
import json
import re

from deref.common import read_jsonl
from deref.inputs import evidence as EV
from deref.inputs import expected_value_parser as EVP
from deref.inputs import state as ST
from deref.inputs.plain import base_parts

HEAD = "\nExpected value (written before seeing the proposed value):\n"
HEAD_R = "\nReference call (revised from the proposed call):\n"
E_DEF = "Expected value for {name}: {value}."
E_UND = "The context does not determine the value of {name}."
SNIPPETS = "Source snippets:\n"
DEFINITE, UNDETERMINED = "definite", "undetermined"

# method -> (item-id tag, body kind, header)
METHODS = {
    "deref": ("deref", "answer", HEAD),
    "sgv_prior": ("sgv_prior", "answer", HEAD),
    "revise": ("revise", "json", HEAD_R),
    "evidence_search": ("evidence_search", "snippets", None),
    "deref_nothink": ("deref_nothink", "output", HEAD),
    "deref_qwen3_8b": ("deref_qwen3_8b", "answer", HEAD),
    "expected_value": ("expected_value", "expected", HEAD),
}


def answer(output):
    """Text after the last </think>, stripped; None if there is no </think> or nothing after it."""
    if "</think>" not in output:
        return None
    a = output.rsplit("</think>", 1)[1].strip()
    return a or None


def parse_block(g):
    """(body, parsed dict or None): the JSON object of a Twin / Revise answer (as is, inside a ```json fence, or from
    the first '{' to the last '}'), re-serialised; the raw text if no JSON object is found."""
    g = g.strip()
    cands = [g]
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", g, flags=re.S)
    if m:
        cands.append(m.group(1))
    if "{" in g and "}" in g:
        cands.append(g[g.index("{"):g.rindex("}") + 1])
    for c in cands:
        try:
            o = json.loads(c)
        except (ValueError, TypeError):
            continue
        if isinstance(o, dict):
            return json.dumps(o, ensure_ascii=False), o
    return g, None


def expected_value_body(name, ref):
    """(slot type, block body) of the expected-value-only input from a DeRef answer `ref`."""
    r = EVP.prior_value12(ref.strip())
    if r["definite"]:
        return DEFINITE, E_DEF.format(name=name, value=r["text"])
    return UNDETERMINED, E_UND.format(name=name)


def insert_block(rec, arg, block, tok, tag):
    """State (re-laddered from the Plain rung) + block + argument tail; first rung with <= 2,048 CLM tokens, else None.
    `block` starts with its own newline (HEAD...) or is preceded by one (snippets)."""
    S, lab, tail, q = base_parts(rec, arg)
    for lab2, S2 in ST.further(ST.window(rec), tok, lab):
        ctx = S2 + block + tail
        n = ST.clm_tokens(tok, ctx, q)
        if n <= ST.LIMIT:
            return {"item_id": f"{arg['item_id']}|{tag}", "base_item": arg["item_id"], "context": ctx, "question": q,
                    "clm_tokens": n, "truncation": lab2, "extra_truncation": lab2 != lab}
    return None


def block_of(method, rec, arg, output):
    """(block text or None, extra fields) for one argument from its generation output."""
    _, kind, head = METHODS[method]
    if kind == "output":
        return head + output.strip(), {}
    a = answer(output)
    if a is None:
        return None, {"reason": "no answer"}
    if kind == "answer":
        return head + a.strip(), {}
    if kind == "json":
        return head + parse_block(a)[0].strip(), {}
    if kind == "expected":
        typ, body = expected_value_body(arg["name"], a)
        return head + body.strip(), {"slot_type": typ}
    sn = EV.evidence_snippets(rec["dataset"], a, ST.pieces(rec))
    if not sn:
        return None, {"reason": "no snippet"}
    return "\n" + SNIPPETS + "\n".join(sn), {"snippets": sn}


def build(method, records, generations, item_map, tok, analysis_only=True):
    """(rows, fallbacks): rows for the arguments mapped to a generation; fallbacks = [(item_id, reason)] for
    arguments that keep their Plain input. generations {prompt_id: output}, item_map {argument item_id: prompt_id}."""
    tag = METHODS[method][0]
    rows, fallbacks = [], []
    for rec, arg in ST.arguments(records, analysis_only):
        pid = item_map.get(arg["item_id"])
        if pid is None:
            continue
        out = generations.get(pid)
        if out is None:
            fallbacks.append((arg["item_id"], "not generated"))
            continue
        block, extra = block_of(method, rec, arg, out)
        if block is None:
            fallbacks.append((arg["item_id"], extra["reason"]))
            continue
        r = insert_block(rec, arg, block, tok, tag)
        if r is None:
            fallbacks.append((arg["item_id"], "over 2048 at every rung"))
            continue
        r.update(extra)
        rows.append(r)
    return rows, fallbacks


def deref_full_block(output):
    """Block of the DeRef + full-history input (None if the DeRef generation has no answer)."""
    a = answer(output)
    return None if a is None else HEAD + a.strip()


def load_generations(path):
    """{prompt_id: output} from a generations JSONL file."""
    return {r["prompt_id"]: r["output"] for r in read_jsonl(path)}


def load_map(path):
    """{argument item_id: prompt_id} from a prompt map JSONL file."""
    return {r["item_id"]: r["prompt_id"] for r in read_jsonl(path)}
