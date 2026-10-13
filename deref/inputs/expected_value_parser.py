"""Expected-value parser v1.2: reads the value a derivation states for the parameter (used by the
expected-value-only ablation, which keeps only that value from the DeRef reference).

v1.1 (`prior_value`): three sentence patterns ("should be [set to | either] V", "default value [of | is | , | :] V",
"V should be used"), earliest match wins; "should be one of" / "one of the" before the value -> undetermined; one layer
of matching quotes (" ' `) stripped. A value token V is a quoted string, a code span, a bracketed list, a number,
true, false or null.
v1.2 (`prior_value12`) = v1.1 with
  (1) normalisation before reading the value: Markdown emphasis markers `**` and `__` are removed (format only);
  (2) an emphasised span (`**…**` / `__…__`) that starts where the value token starts is one value token as a whole
      (e.g. `The value should be **2024-11-28**.` -> 2024-11-28);
  (3) three fallback value patterns (P4 default phrasing, P5 "should be the … i.e. / which is value", P6 "should use
      `name` = value"), used only when none of the v1.1 sentence patterns matches;
  (4) a veto for a second candidate value in the same sentence (see `multi_candidate`).
Per reference, prior_value12 returns definite, value (v1.1 semantics, outer quotes stripped), text (the value as written,
without emphasis markers, adjacent same-kind quotes kept — used by the expected-value block), core (value without any
quote layers), core_span / text_span (start, end) in the original reference text."""
import re

VERSION = "v1.2"

# ---------------------------------------------------------------- v1.1
VTOK = r"(\"[^\"]*\"|'[^']*'|`[^`]*`|\[[^\]]*\]|-?\d+(?:\.\d+)?|true|false|null)"
PATS = [re.compile(r"should be(?: set to| either)?\s+" + VTOK, re.I),
        re.compile(r"default value(?:\s+of|\s+is|,|:)?\s+" + VTOK, re.I),
        re.compile(VTOK + r"\s+should be used", re.I)]


def prior_value(t):
    """v1.1: (definite, value text or None)."""
    best = None
    for p in PATS:
        m = p.search(t)
        if m and (best is None or m.start() < best.start()):
            best = m
    if best is None:
        return False, None
    pre = t[:best.end()].lower()
    if "should be one of" in pre or "one of the" in pre:
        return False, None
    v = best.group(1)
    if v[:1] in "\"'`" and v[-1:] == v[:1]:
        v = v[1:-1]
    return True, v


# ---------------------------------------------------------------- v1.2
EMPH = re.compile(r"(\*\*)(?=\S)(.+?)(?<=\S)\*\*|(?<![A-Za-z0-9_])(__)(?=\S)(.+?)(?<=\S)__(?![A-Za-z0-9_])")
OPEN, CLOSE = "\ue000", "\ue001"  # private-use sentinels marking emphasised spans inside the normalised text
QUOTES = "\"'`"
VTOK12 = OPEN + r"[^" + CLOSE + r"]*" + CLOSE + "|" + VTOK[1:-1]
PATS12 = [re.compile(p.pattern.replace(VTOK, "(" + VTOK12 + ")"), re.I) for p in PATS]
assert all(VTOK in p.pattern for p in PATS) and len(PATS12) == 3
# Fallback value patterns, used only when none of the three v1.1 sentence patterns matches (so every value v1.1 reads
# is read unchanged); earliest match among them, same "one of" veto.
V_ = "(" + VTOK12 + ")"
FALLBACK = [
    # P4 default phrasing, e.g. `should be its default, `"dontcare"`` -> dontcare; `would take its stated default of 2` -> 2
    re.compile(r"default(?: value)?(?:,|:|\s+of|\s+is|,?\s+which is:?)?\s+" + V_, re.I),
    # P5 descriptive "should be the …" followed in the same sentence by `i.e.` / `which is` and the value, e.g.
    #   `should be the `search_key` field returned by the call in step 1, which is: `"eyJ…"`` -> eyJ…
    #   The descriptive part takes code spans `…` whole; a dot followed by a non-space (`data.search_key`,
    #   `flightOffers[0].token`) does not end the sentence.
    re.compile(r"should be(?: set to)?\s+(?:the|a|an)\s(?:[^.\n`]|\.(?=\S)|`[^`\n]*`){0,200}?"
               r"(?:,?\s*i\.e\.,?|,?\s*which is:?)\s+(?:something like\s+)?" + V_, re.I),
    # P6 `should use `name` = value`, e.g. `should use `symbol` = **"MSFT"**` -> MSFT
    re.compile(r"should use\s+`?[\w.]+`?\s*=\s*" + V_, re.I),
]


def normalise(t):
    """(text with markers removed and emphasised spans wrapped in sentinels, index map normalised -> original;
    a sentinel maps to the position of the marker it replaces). `__` counts as emphasis only Markdown-style (not inside
    a word, so identifiers such as __typename are untouched); an unpaired `**` is removed without a sentinel."""
    out, idx = [], []
    i = 0
    for m in EMPH.finditer(t):
        a, b, w = m.start(), m.end(), 2
        out += t[i:a]; idx += range(i, a)
        out.append(OPEN); idx.append(a)
        out += t[a + w:b - w]; idx += range(a + w, b - w)
        out.append(CLOSE); idx.append(b - w)
        i = b
    out += t[i:]; idx += range(i, len(t))
    s = "".join(out)
    keep, k = [], 0
    while k < len(s):
        if s.startswith("**", k):
            k += 2
            continue
        keep.append(k)
        k += 1
    return "".join(s[k] for k in keep), [idx[k] for k in keep]


def plain(s):
    return s.replace(OPEN, "").replace(CLOSE, "")


# Veto for a second candidate value in the same sentence (narrow). Applies to every pattern, v1.1 ones included.
#   (a) "or": right after the value (one complete parenthetical may sit between) comes `or VALUE` / `, or VALUE`
#       (covers "either X or Y"); an "or" inside parentheses ("(or equivalently "annual")") does not count.
#   (b) per situation: in the sentence, values attached to a situation — `for A [it] should be / would be / is / = / : VALUE`
#       (the linking word is required: `for 2009` alone is not a value) or `VALUE (for A)` — at least two different ones.
#   Not counted: anything inside parentheses (restatements; (b)'s `VALUE (for A)` keeps its value outside), text after
#   `i.e.`, quoted source text ("…" with 3+ spaces inside, quotes paired with code spans hidden), and negations
#   (`not Y`, `instead of Y`, `rather than Y` match neither form). A value in a `for …` clause must end at a word
#   boundary (`for the 60-day …` is not the value 60).
OR_NEXT = re.compile(r"\s*(?:\([^()]*\)\s*)?,?\s*or\s+" + V_, re.I)
FOR_V = re.compile(r"\bfor\s+[^,;()\n]{1,60}?,?\s*(?:it\s+)?(?:should be|would be|is|=|:)\s*" + V_ + r"(?![\w\-/:])", re.I)
V_FOR = re.compile(V_ + r"\s*\(for\s+[^()]{1,60}\)", re.I)
SENT_END = re.compile(r"[.!?][\"'*\u201d]*(?=\s|$)|\n")


def _canon(tok):
    x = plain(tok).strip()
    while len(x) >= 2 and x[0] in QUOTES and x[-1] == x[0]:
        x = x[1:-1]
    return " ".join(x.lower().split())


def _mask(seg):
    """blank out parentheticals, text after i.e., and quoted source text (same length, so positions are kept)."""
    out = list(seg)
    depth = 0
    for i, ch in enumerate(seg):
        if ch == "(":
            depth += 1
        if depth > 0:
            out[i] = " "
        if ch == ")" and depth > 0:
            depth -= 1
    m = re.search(r"\bi\.e\.", seg)
    if m:
        out[m.start():] = " " * (len(seg) - m.start())
    # quoted source text: double-quoted spans with 3+ spaces inside; code spans `…` are hidden while pairing the quotes
    hid = re.sub(r"`[^`\n]*`", lambda m: "#" * len(m.group(0)), seg)
    for m in re.finditer(r"\"[^\"\n]*\"", hid):
        if m.group(0).count(" ") >= 3:
            out[m.start():m.end()] = " " * (m.end() - m.start())
    return "".join(out)


def multi_candidate(n, s, e):
    """True when the sentence of the value at n[s:e] gives a second candidate value (rules (a), (b) above)."""
    m = OR_NEXT.match(n, e)
    if m and _canon(m.group(1)) != _canon(n[s:e]):
        return True
    a = max([x.end() for x in SENT_END.finditer(n, 0, s)] or [0])
    z = SENT_END.search(n, e)
    z = z.start() if z else len(n)
    seg = n[a:z]
    vals = set()
    for x in V_FOR.finditer(seg):  # the "(for A)" part is masked later, so use the raw sentence here
        if "(" not in _mask_keep_paren_start(seg, x.start()):
            vals.add(_canon(x.group(1)))
    ms = _mask(seg)
    for x in FOR_V.finditer(ms):
        vals.add(_canon(x.group(1)))
    return len(vals) >= 2


def _mask_keep_paren_start(seg, i):
    """'(' when position i lies inside parentheses, else ''."""
    depth = 0
    for ch in seg[:i]:
        depth += ch == "("
        depth -= ch == ")" and depth > 0
    return "(" if depth > 0 else ""


def prior_value12(t, veto=True):
    n, idx = normalise(t)
    best = None
    for pats in (PATS12, FALLBACK):
        for p in pats:
            m = p.search(n)
            if m and (best is None or m.start() < best.start()):
                best = m
        if best is not None:
            break
    nul = {"definite": False, "value": None, "text": None, "core": None, "core_span": None, "text_span": None}
    if best is None:
        return nul
    pre = plain(n[:best.end()]).lower()
    if "should be one of" in pre or "one of the" in pre:
        return nul
    s, e = best.span(1)
    if veto and multi_candidate(n, s, e):
        return nul | {"vetoed": True}
    if n[s] == OPEN:  # emphasised span: its inside is the token
        s, e = s + 1, e - 1
    tok = n[s:e]
    text = plain(tok).strip()
    if not text:
        return nul
    # positions of the stripped text inside n (ignoring sentinels)
    ks = [k for k in range(s, e) if n[k] not in (OPEN, CLOSE)]
    lead = len(plain(tok)) - len(plain(tok).lstrip())
    ks = ks[lead:lead + len(text)]
    val = text[1:-1] if (text[:1] in QUOTES and text[-1:] == text[:1] and len(text) >= 2) else text
    core, a, b = text, 0, len(text)
    while len(core) >= 2 and core[0] in QUOTES and core[-1] == core[0]:
        core, a, b = core[1:-1], a + 1, b - 1
    text_span = (idx[ks[0]], idx[ks[-1]] + 1)
    core_span = (idx[ks[a]], idx[ks[b - 1]] + 1) if b > a else (idx[ks[a - 1]] + 1,) * 2  # empty value, e.g. `""`
    return {"definite": True, "value": val, "text": text, "core": core, "core_span": core_span, "text_span": text_span}
