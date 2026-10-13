"""Derivation path of every labelled argument (string-matching categories, not the agent's actual derivation).

For an argument value v of a call at a state, checked in this order:
  placeholder   v is null, "", [], {} or a (non-boolean) number equal to 0;
  visible       v is found in the CONTENT of the final state (`content_text`: question text, shown call texts, shown
                return texts; the fixed template text — "Question: ", the header line, "- ", "  -> ", "none" — removed);
  history_only  v is found in the full untruncated history (question + every earlier call's arguments + every return);
  format        a normalised match in the history pieces: case / whitespace, same date (several formats), same number
                (commas removed), same boolean; lists: every element found or matched;
  default       v equals the schema default (strings case-insensitive), or is "dontcare" / "don't care" / "any" quoted
                in the parameter description, or "use X to indicate / for dontcare" in the description;
  no_source     otherwise.
`found`: strings compared as is, lists / dicts as compact sorted JSON, other values as JSON; a list is found if every
element is found; word boundaries for non-strings and strings shorter than 4 characters; numbers (and strings that are
one number) need no letter / digit / "." before and no letter / digit / ".<digit>" after.
Internal codes: PH placeholder, P1 visible, P2 history_only, P3 format, P4 default, P5 no_source.
"""
import datetime as dt
import json
import re

from deref.build.items import read_jsonl

PATH_NAMES = {"P1": "visible", "P2": "history_only", "P3": "format", "P4": "default", "P5": "no_source", "PH": "placeholder"}


# ---------------------------------------------------------------- found
def norm(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (list, dict)):
        return json.dumps(v, sort_keys=True, separators=(",", ":"))
    return json.dumps(v)


def contains(s, text, boundary, numeric=False):
    if numeric:  # no letter / digit / "." before, no letter / digit / ".<digit>" after
        return re.search(r"(?<![A-Za-z0-9.])" + re.escape(s) + r"(?![A-Za-z0-9])(?!\.\d)", text) is not None
    if not boundary:
        return s in text
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(s) + r"(?![A-Za-z0-9])", text) is not None


def numeric(v):
    """A number (not bool) or a string that is one number."""
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    return isinstance(v, str) and re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", v.strip()) is not None


def placeholder(v):
    """0, "", null, [] or {}."""
    return v is None or v == "" or v == [] or v == {} or (not isinstance(v, bool) and isinstance(v, (int, float)) and v == 0)


def found(v, text):
    if v is None or v == "" or v == [] or v == {}:
        return False
    if isinstance(v, list) and all(found(x, text) for x in v):
        return True
    s = norm(v)
    return contains(s, text, not isinstance(v, str) or len(s) < 4, numeric(v))


# ---------------------------------------------------------------- format (normalised) matches
MON = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"])}
MON.update({k[:3]: v for k, v in list(MON.items())})
MON["sept"] = 9
MRE = "(" + "|".join(sorted(MON, key=len, reverse=True)) + r")\.?"
ORD = r"(?:st|nd|rd|th)?"
DATE_PATS = [  # (regex, order of groups: y m d), month group may be a name
    (r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", "ymd"),
    (r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)", "mdy"),
    (r"(?<![A-Za-z])" + MRE + r"\s+(\d{1,2})" + ORD + r",?\s+(\d{4})(?!\d)", "mdy"),
    (r"(?<!\d)(\d{1,2})" + ORD + r"(?:\s+of)?\s+" + MRE + r",?\s+(\d{4})(?!\d)", "dmy"),
]
TIME = r"(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"


def mkdate(g, order):
    d = dict(zip(order, g))
    m = MON.get(d["m"].lower()) if not d["m"].isdigit() else int(d["m"])
    try:
        return dt.date(int(d["y"]), m, int(d["d"]))
    except (TypeError, ValueError):
        return None


def text_dates(text):
    out = set()
    for p, order in DATE_PATS:
        for g in re.findall(p, text, flags=re.I):
            x = mkdate(g, order)
            if x:
                out.add(x)
    return out


def value_date(v):
    if not isinstance(v, str):
        return None
    s = v.strip()
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", s)
    if m:
        return mkdate(m.groups(), "ymd")
    for p, order in DATE_PATS:
        m = re.fullmatch(p + TIME, s, flags=re.I)
        if m:
            return mkdate(m.groups(), order)
    return None


NUM_TXT = r"(?<![A-Za-z0-9.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![0-9])"


def text_numbers(text):
    return {float(x.replace(",", "")) for x in re.findall(NUM_TXT, text)}


def value_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", v.strip()):
        return float(v.strip().replace(",", ""))
    return None


def value_bool(v):
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower()
    return None


def lowws(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


class Ctx:
    """question + full history pieces of one state, with the normalised-match indexes."""

    def __init__(self, pieces):
        self.low = [lowws(t) for t in pieces]
        joined = "\n".join(pieces)
        self.dates, self.nums = text_dates(joined), text_numbers(joined)
        self.bools = {b for b in ("true", "false") if re.search(rf"\b{b}\b", joined, flags=re.I)}


def p3_scalar(v, ctx):
    if v is None or v == "" or v == [] or v == {}:
        return None
    s = lowws(norm(v))
    if s and any(contains(s, t, len(s) < 4, numeric(v)) for t in ctx.low):
        return "case / whitespace"
    d = value_date(v)
    if d and d in ctx.dates:
        return "date"
    n = value_number(v)
    if n is not None and n in ctx.nums:
        return "number"
    b = value_bool(v)
    if b and b in ctx.bools:
        return "boolean"
    return None


def p3(v, ctx, hist_text):
    if isinstance(v, list) and v:
        kinds = []
        for x in v:
            if found(x, hist_text):
                continue
            k = p3_scalar(x, ctx)
            if k is None:
                return None
            kinds.append(k)
        return "list: " + "+".join(sorted(set(kinds))) if kinds else "list"
    return p3_scalar(v, ctx)


# ---------------------------------------------------------------- default
def is_default(v, spec):
    if not spec:
        return None
    if "default" in spec:
        d = spec["default"]
        if v == d:
            return "schema default"
        if isinstance(v, str) and isinstance(d, str) and v.lower() == d.lower():
            return "schema default"
    desc = spec.get("description") or ""
    if isinstance(v, str) and v.lower() in ("dontcare", "don't care", "any"):
        if re.search(r"""['"`]""" + re.escape(v) + r"""['"`]""", desc, flags=re.I):
            return "placeholder in description"
    for x in re.findall(r"""use\s+['"`]?([^\s'"`]+)['"`]?\s+(?:to indicate|for)\s+['"`]?dontcare""", desc, flags=re.I):
        if norm(v) == x:
            return "placeholder in description"
    return None


# ---------------------------------------------------------------- census
def content_text(state, header):
    """The state's content only: question text, call texts, shown return texts, user texts (template text removed)."""
    assert state.startswith("Question: ")
    body = state[len("Question: "):]
    key = "\n" + header + "\n"
    if key in body:
        q, rest = body.split(key, 1)
    else:
        m = re.search(r"\nPrevious calls[^\n]*:\n", body)
        assert m, header
        q, rest = body[:m.start()], body[m.end():]
    out = [q]
    if rest != "none":
        for ln in rest.split("\n"):
            for pre in ("- ", "  -> ", "User: "):
                if ln.startswith(pre):
                    ln = ln[len(pre):]
                    break
            out.append(ln)
    return "\n".join(out)


def census(export_path, states, tools_path, template_free=True):
    """{item_id: internal path code} for every argument labelled correct / wrong in the export.
    states: {pos_id: final state text}; tools_path: FinTrace tools.jsonl (parameter specs for `default`).
    template_free=False searches `visible` in the whole state text (earlier rule version, kept for checking)."""
    tools = {}
    for r in read_jsonl(tools_path):
        tools[r["endpoint"]] = {p["name"]: p for p in r["params"]}
    out = {}
    with open(export_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            pieces = [t for _, t in s["history"]]
            hist, ctx = "\n".join(pieces), None
            st = content_text(states[s["pos_id"]], s["header"]) if template_free else states[s["pos_id"]]
            for c in s["calls"]:
                for it in c["items"]:
                    if it["kind"] != "param" or it["label"] not in ("correct", "wrong"):
                        continue
                    fn = c["parsed"][it["call_index"]][0]
                    if s["dataset"] == "FinTrace":
                        sp = tools.get(fn, {}).get(it["param"])
                    else:
                        sp = (c["schemas"][it["call_index"]] if c["schemas"] else {}).get(it["param"])
                    v = it["value"]
                    if placeholder(v):
                        path = "PH"
                    elif found(v, st):
                        path = "P1"
                    elif found(v, hist):
                        path = "P2"
                    else:
                        ctx = ctx or Ctx(pieces)
                        path = "P3" if p3(v, ctx, hist) else "P4" if is_default(v, sp) else "P5"
                    out[it["item_id"]] = path
    return out
