"""Evidence search: parsing the quoted source text out of a generation, verification against the history pieces
(strict whitespace-insensitive match, then a lenient longest-common-substring match), and the snippet rules.

Snippet rules: ComplexFuncBench ±100 characters, no cap (`snippets_cfb`); FC-RewardBench and FinTrace ±150 characters,
cap 450 (`snippets_capped`, given occurrence spans)."""
import difflib
import re

SNIP_CFB, MERGE_GAP, MAXOCC = 100, 50, 3
# capped snippet format (FC-RewardBench, FinTrace): ±150 characters, merge gap 50, cap 450, at most 3 snippets
SNIP, SNIP_MAX = 150, 450
PAIRS = {'"': '"', "'": "'", "\u201c": "\u201d", "\u2018": "\u2019", "`": "`"}
LABEL = re.compile(r"^(?:Question:|\[(?:call|return), step \d+\]|\[question\])\s*")
UNESC = re.compile(r'\\(["\\])')  # \" -> " and \\ -> \ in one pass


def step_of(lab):
    return int(lab.rsplit("step ", 1)[1])


def qpattern(q):
    return r"\s+".join(re.escape(w) for w in q.split())


def unquote(q):
    """One layer of matching quotation marks around the whole quote removed."""
    if len(q) >= 2 and PAIRS.get(q[0]) == q[-1]:
        return q[1:-1].strip()
    return q


def parse(output):
    """('NONE', []) or ('quotes', [quote lines])."""
    o = output.strip()
    if re.fullmatch(r"(?i)none\.?", o):
        return "NONE", []
    lines = [unquote(x.strip()) for x in o.split("\n")]
    lines = [x for x in lines if x and not re.fullmatch(r"(?i)none\.?", x)]
    return ("quotes" if lines else "NONE"), lines


def forms(q):
    """Quote forms tried in order: label prefix removed (and quotation marks once more), then JSON-unescaped."""
    q = unquote(LABEL.sub("", q, count=1).strip())
    u = UNESC.sub(r"\1", q)
    return [q] + ([u] if u != q else [])


def verify(quotes, pieces):
    """[(quote, verified, {piece index: [(start, length)]})] for the first 2 quotes."""
    out = []
    for q in quotes[:2]:
        occ = {}
        for f_ in forms(q):
            if not f_.strip():
                continue
            pat = qpattern(f_)
            occ = {i: f for i, (_, t) in enumerate(pieces) if (f := [(m.start(), m.end() - m.start()) for m in re.finditer(pat, t)])}
            if occ:
                break
        out.append((q, bool(occ), occ))
    return out


def lcs(a, b):
    """(length, start in a) of the longest common substring of a (long) and b (short)."""
    m = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size, m.a


def verify2(quotes, pieces):
    """Strict and lenient occurrences: (strict_occ, lenient_occ, per-quote [(quote, strict ok, lenient ok)]).
    Lenient: an unverified quote counts when its longest common substring with some piece has at least 30 characters
    and at least 60% of the quote."""
    so, lo, per = {}, {}, []
    for q, ok, occ in verify(quotes, pieces):
        lok = ok
        for i, sp in occ.items():
            so.setdefault(i, []).extend(sp)
            lo.setdefault(i, []).extend(sp)
        if not ok:
            f0 = forms(q)[0] if forms(q) else q
            if f0.strip():
                best = max(((*lcs(t, f0), i) for i, (_, t) in enumerate(pieces)), key=lambda x: (x[0], -x[2]))
                n, st, i = best
                if n >= 30 and n >= 0.6 * len(f0):
                    lo.setdefault(i, []).append((st, n))
                    lok = True
        per.append((q, ok, lok))
    return so, lo, per


def snippets_cfb(occ, pieces):
    """±100 characters, merge < 50 apart, no cap, at most 3, most recent pieces first."""
    cand = []
    for idx in reversed(range(len(pieces))):
        if idx not in occ:
            continue
        lab, txt = pieces[idx]
        spans = []
        for i, L in sorted(occ[idx]):
            a, b = max(0, i - SNIP_CFB), min(len(txt), i + L + SNIP_CFB)
            if spans and a - spans[-1][1] < MERGE_GAP:
                spans[-1][1] = max(spans[-1][1], b)
            else:
                spans.append([a, b])
        for a, b in spans:
            cand.append((lab, f"…{txt[a:b]}…"))
    merged = {}
    for lab, body in cand:
        if body not in merged and len(merged) == MAXOCC:
            continue
        merged.setdefault(body, []).append(lab)
    out = []
    for body, labs in merged.items():
        labs = sorted(labs, key=lambda l: step_of(l) if ", step" in l else -1)
        kinds = {l.split(", step")[0] for l in labs}
        if len(kinds) == 1 and all(", step" in l for l in labs):
            label = labs[0].split(", step")[0] + ", " + "; ".join("step" + l.split(", step", 1)[1] for l in labs)
        else:
            label = "; ".join(labs)
        out.append(f"[{label}] {body}")
    return out


def snippets_capped(occ, pieces):
    """Value-search snippet format with given occurrence spans {piece index: [(start, length)]}: ±150, cap 450."""
    cand = []
    for idx in reversed(range(len(pieces))):
        if idx not in occ:
            continue
        lab, txt = pieces[idx]
        spans = []
        for i, L in sorted(occ[idx]):
            a, b = max(0, i - SNIP), min(len(txt), i + L + SNIP)
            if spans and a - spans[-1][1] < MERGE_GAP:
                spans[-1][1] = max(spans[-1][1], b)
                spans[-1][3] = max(spans[-1][3], i + L)
            else:
                spans.append([a, b, i, i + L])
        for a, b, c0, c1 in spans:
            if b - a > SNIP_MAX:
                mid = (c0 + c1) // 2
                a2 = max(0, min(mid - SNIP_MAX // 2, len(txt) - SNIP_MAX))
                cand.append((lab, f"...{txt[a2:a2 + SNIP_MAX]}..."))
            else:
                cand.append((lab, f"…{txt[a:b]}…"))
    merged = {}
    for lab, body in cand:
        if body not in merged and len(merged) == MAXOCC:
            continue
        merged.setdefault(body, []).append(lab)
    out = []
    for body, labs in merged.items():
        labs = sorted(labs, key=lambda l: int(l.rsplit("step ", 1)[1]) if ", step" in l else -1)
        kinds = {l.split(", step")[0] for l in labs}
        if len(kinds) == 1 and all(", step" in l for l in labs):
            label = labs[0].split(", step")[0] + ", " + "; ".join("step" + l.split(", step", 1)[1] for l in labs)
        else:
            label = "; ".join(labs)
        out.append(f"[{label}] {body}")
    return out


def evidence_snippets(dataset, answer_text, pieces):
    """Snippet lines for an evidence-search answer (text after </think>); [] = no snippet (the Plain input is kept)."""
    if answer_text is None:
        return []
    kind, quotes = parse(answer_text)
    if kind != "quotes":
        return []
    _, lo, _ = verify2(quotes, pieces)
    if not lo:
        return []
    return snippets_cfb(lo, pieces) if dataset == "cfb" else snippets_capped(lo, pieces)
