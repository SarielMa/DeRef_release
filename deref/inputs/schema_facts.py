"""Schema-facts judge inputs (Figure 2b): the Plain input of an FC-RewardBench argument whose schema has a default,
plus one line that restates the default and lists which related words occur in the request.

Line: 'Schema default for <param>: <default JSON>.' followed by either
  ' The question contains: "<snippet>", "<snippet>", ...'  (one snippet per related word found), or
  ' The question does not contain any of these words: "<word>", ...'.
Related words: the parameter name, the name with "_" replaced by " ", the default (as text), every enum value, and
"true" / "false" for a boolean parameter; duplicates (case-insensitive) are dropped, first occurrence kept.
A word is found at its first case-insensitive match not adjacent to a letter or digit; its snippet is the request text
40 characters around the match, whitespace collapsed, double quotes replaced by single quotes, with "…" where cut.
Only the judge input changes; the line is appended after the Plain input (after the proposed-value line).
Item id = "<argument item_id>|schema_facts"."""
import json
import re

from deref.inputs import state as ST
from deref.inputs.plain import plain_item

TAG = "schema_facts"


def _text(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def related_words(param, spec):
    w = [param, param.replace("_", " "), _text(spec["default"])]
    if "enum" in spec:
        w += [_text(x) for x in spec["enum"]]
    if spec.get("type") == "boolean":
        w += ["true", "false"]
    out, seen = [], set()
    for x in w:
        if x and x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out


def snippet(q, a, b):
    lo, hi = max(0, a - 40), min(len(q), b + 40)
    s = " ".join(q[lo:hi].split()).replace('"', "'")
    return ("…" if lo > 0 else "") + s + ("…" if hi < len(q) else "")


def schema_facts_line(param, spec, request):
    """(line, hit): the schema-facts line of one argument and whether any related word occurs in the request."""
    words = related_words(param, spec)
    hits = []
    for w in words:
        m = re.search(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", request, re.I)
        if m:
            hits.append(snippet(request, *m.span()))
    first = f"Schema default for {param}: {json.dumps(spec['default'], ensure_ascii=False)}."
    if hits:
        return first + " The question contains: " + ", ".join(f'"{h}"' for h in hits) + ".", True
    return first + " The question does not contain any of these words: " + ", ".join(f'"{w}"' for w in words) + ".", False


def build(records, analysis_only=True):
    """Schema-facts rows of the FC-RewardBench arguments whose per-call schema has a `default` field."""
    rows = []
    for rec, arg in ST.arguments(records, analysis_only):
        assert rec["dataset"] == "fcrb", "schema-facts inputs are defined for FC-RewardBench only"
        spec = ST.call_schemas(rec)[arg["call_index"]].get(arg["name"]) or {}
        if "default" not in spec:
            continue
        line, hit = schema_facts_line(arg["name"], spec, rec["request"])
        p = plain_item(rec, arg)
        rows.append({"item_id": f"{arg['item_id']}|{TAG}", "base_item": arg["item_id"], "context": p["context"] + "\n" + line,
                     "question": p["question"], "label": arg["label"], "path": arg["path"], "hit": hit,
                     "value": arg["value"], "default": spec["default"]})
    return rows
