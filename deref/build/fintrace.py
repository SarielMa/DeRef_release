"""FinTrace: test set + FMP docs -> tool catalog -> endpoint similarity -> correct calls and constructed wrong calls.

Sources
  test set:  HF `YupengCao/FinTrace`, file evaluation/testset.json (800 queries). The HF revision was not recorded;
             the file is pinned by TESTSET_SHA256.
  FMP docs:  the public FMP stable API docs pages (https://site.financialmodelingprep.com/developer/docs), scraped
             once by `fetch_fmp_docs` (live pages; a re-fetch is not guaranteed to give the same text). The build reads
             the cached pages.
Randomness: every draw uses random.Random(f"{SEED}|<key>") with a stable key, so results do not depend on order.
Wrong calls: P4 = one argument error per correct call (ticker / date / period), P2_bin1..5 = endpoint substitution by
similarity bin (used only as further calls of the same state when the 2,048-token rule is applied).
"""
import collections as C
import datetime as dt
import html
import json
import random
import re
import time
import urllib.request

from deref.build.items import NOT_PARAM, call_items, read_jsonl, reword_not_param, split_call, write_items, write_jsonl

SEED = 20260930
TESTSET_SHA256 = "b10f3edcfe5df13b9d3aa9cc4dccb90dcc523234ee0c19cee5cd12c68055eb0a"
BGE = "BAAI/bge-large-en-v1.5"
BGE_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"


def load_testset(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def golden_calls(example):
    """Expand golden tool calls to endpoint level.

    Each item: {turn, call_idx, category, endpoint, args (without 'endpoint'), raw_args, result}.
    `turn` indexes assistant messages that carry tool_calls; results are joined via tool_call_id.
    A few calls are named "category:endpoint" (with or without an endpoint arg); the name is split on ':'.
    """
    msgs = example["golden_trajectories"]
    results = {m.get("tool_call_id"): m.get("content", "") for m in msgs if m.get("role") == "tool"}
    calls, turn = [], -1
    for m in msgs:
        tcs = m.get("tool_calls") or [] if m.get("role") == "assistant" else []
        if not tcs:
            continue
        turn += 1
        for tc in tcs:
            f = tc["function"]
            raw = json.loads(f.get("arguments") or "{}")
            args = {k: v for k, v in raw.items() if k != "endpoint"}
            category, _, name_ep = f["name"].partition(":")
            endpoint = raw.get("endpoint") or name_ep or None
            calls.append({"turn": turn, "call_idx": len(calls), "category": category, "endpoint": endpoint,
                          "args": args, "raw_args": f.get("arguments"), "result": results.get(tc.get("id"))})
    return calls


# ---------------------------------------------------------------- FMP docs scrape (network)
DOCS = "https://site.financialmodelingprep.com/developer/docs"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
KEEP = ("title", "pageURL", "description", "content", "params", "urls", "code", "subcode", "id", "schemaName", "status")


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def _next_data(page):
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', page, re.S)
    return json.loads(m.group(1))["props"]["pageProps"]


def fetch_fmp_docs(out_dir):
    """Cache the docs index (_index.json) and each endpoint page's `doc` record (<slug>.json); skips cached pages."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index_html = _get(DOCS)
    slugs = sorted(set(re.findall(r'href="/developer/docs/stable/([^"#?]+)', index_html)))
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    (out_dir / "_index.json").write_text(json.dumps({"url": DOCS, "fetched_at_utc": fetched_at, "slugs": slugs}, indent=1))
    failed = []
    for i, slug in enumerate(slugs):
        out = out_dir / f"{slug}.json"
        if out.exists():
            continue
        url = f"{DOCS}/stable/{slug}"
        try:
            doc = _next_data(_get(url)).get("doc") or {}
        except Exception as e:  # noqa: BLE001
            failed.append((slug, f"{type(e).__name__}: {e}"))
            continue
        rec = {"source_url": url, "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "doc": {k: doc.get(k) for k in KEEP}}
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(1.0)
        if i % 25 == 0:
            print(f"{i}/{len(slugs)}", flush=True)
    print(f"slugs={len(slugs)} failed={len(failed)}")
    for f in failed:
        print("FAILED", *f)


# ---------------------------------------------------------------- tool catalog
# De-naming: self-references such as "the (FMP) Company Profile Data API" are replaced by "this API" ("This API" at a
# sentence start). The description used in the action text and for func_sim is the de-named one.
_NAME_TOKEN = r"(?:\(?[A-Z0-9][\w&/.'’-]*\)?|of|and|for|by|&|to|the|in|on|vs\.?)"
SELF_REF = re.compile(r"(?:\b[Tt]he\s+)?(?:\bFMP(?:'s|’s)?\s+)?(?:" + _NAME_TOKEN + r"\s+){0,10}?(?<![a-z])\(?\b[A-Z0-9][\w&/.'’-]*\)?\s+API\b(?!s)")


def dename(text):
    def rep(m):
        before = text[:m.start()].rstrip()
        return "This API" if not before or before[-1] in ".!?" else "this API"
    return SELF_REF.sub(rep, text)


def strip_html(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def first_paragraph(content):
    m = re.search(r"<p>(.*?)</p>", content or "", re.S)
    return strip_html(m.group(1) if m else content)


def parse_params(params):
    out = []
    for section, table in (params or {}).items():
        for row in table.get("rows", []):
            name = row[0]
            out.append({"name": name.rstrip("*"), "required": name.endswith("*"), "type": row[1] if len(row) > 1 else None,
                        "example": row[2] if len(row) > 2 else None, "in": section})
    return out


def build_tools(testset_path, docs_dir, out_path):
    """tools.jsonl: every endpoint of the docs index. Description = docs `description`, else the first paragraph of
    `content` (HTML stripped), else the page title (desc_source = title_only). Endpoints are matched to golden
    endpoint names case-insensitively."""
    with open(docs_dir / "_index.json", encoding="utf-8") as f:
        index = json.load(f)
    d = load_testset(testset_path)
    golden = C.Counter()
    golden_cat = C.defaultdict(C.Counter)
    for x in d:
        for c in golden_calls(x):
            golden[c["endpoint"]] += 1
            golden_cat[c["endpoint"]][c["category"]] += 1
    golden_by_lower = {e.lower(): e for e in golden}
    rows = []
    for slug in index["slugs"]:
        with open(docs_dir / f"{slug}.json", encoding="utf-8") as f:
            rec = json.load(f)
        doc = rec["doc"]
        desc = strip_html(doc.get("description"))
        desc_field = "description"
        if not desc:
            desc, desc_field = first_paragraph(doc.get("content")), "content_first_paragraph"
        desc_source = "fmp_docs"
        if not desc:  # page exists but has only a title
            desc, desc_field, desc_source = strip_html(doc.get("title")), "title", "title_only"
        ep = golden_by_lower.get(slug.lower(), slug)
        rows.append({
            "endpoint": ep, "docs_slug": slug, "category": doc.get("code"), "title": doc.get("title"),
            "description": dename(desc) if desc_source == "fmp_docs" else desc, "description_raw": desc,
            "desc_field": desc_field,
            "desc_source": desc_source, "source_url": rec["source_url"],
            "fetched_at_utc": rec["fetched_at_utc"], "params": parse_params(doc.get("params")),
            "api_urls": doc.get("urls"), "in_golden": ep in golden, "golden_calls": golden.get(ep, 0),
            "golden_categories": dict(golden_cat.get(ep, {})),
        })
    write_jsonl(out_path, rows)
    print(f"[fintrace tools] catalog {len(rows)}")


# ---------------------------------------------------------------- endpoint similarity
FUNC_QUANTILES = (0.2, 0.4, 0.6, 0.8)  # P2 five bins
P1_FUNC_Q, P3_FUNC_LO_Q, P3_FUNC_HI_Q = 0.5, 0.5, 0.8
P3A_STRICT_NAME_MIN = 0.5


def name_tokens(ep):
    return set(ep.lower().split("-"))


def jaccard(a, b):
    return len(a & b) / len(a | b)


def tool_similarity(tools_path, out_path):
    """tool_sim.npz: name_sim (Jaccard of '-' tokens), func_sim (cosine of normalised bge-large-en-v1.5 embeddings of
    the de-named descriptions, no instruction prefix, CPU), same_category, func_sim bin cut points (quantiles over all
    unordered endpoint pairs) and thresholds. An existing file is kept if the recomputation matches it, else the
    function aborts (the cut points are frozen once written)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    tools = read_jsonl(tools_path)
    eps = [t["endpoint"] for t in tools]
    n = len(eps)
    toks = [name_tokens(e) for e in eps]
    name_sim = np.array([[jaccard(a, b) for b in toks] for a in toks])
    model = SentenceTransformer(BGE, device="cpu", revision=BGE_REVISION)
    emb = model.encode([t["description"] for t in tools], normalize_embeddings=True, batch_size=32)
    func_sim = emb @ emb.T
    cats = np.array([t["category"] for t in tools])
    same_cat = cats[:, None] == cats[None, :]
    iu = np.triu_indices(n, 1)
    fs = func_sim[iu]
    func_cuts = np.quantile(fs, FUNC_QUANTILES)
    thresholds = {
        "p1_func_max": float(np.quantile(fs, P1_FUNC_Q)),
        "p3a_name_gt": 0.0, "p3a_func_max": float(np.quantile(fs, P3_FUNC_LO_Q)),
        "p3a_strict_name_min": P3A_STRICT_NAME_MIN,
        "p3b_name_eq": 0.0, "p3b_func_min": float(np.quantile(fs, P3_FUNC_HI_Q)),
    }
    if out_path.exists():
        old = np.load(out_path)
        same = (list(old["endpoints"]) == eps and np.allclose(old["func_sim"], func_sim, atol=1e-5)
                and np.array_equal(old["name_sim"], name_sim)
                and np.allclose(old["func_bin_cuts"], func_cuts, atol=1e-6)
                and all(abs(json.loads(str(old["thresholds"]))[k] - v) < 1e-6 for k, v in thresholds.items()))
        if not same:
            raise SystemExit(f"{out_path} exists and the recomputed similarities differ; refusing to overwrite.")
        print(f"{out_path} exists and matches the recomputation; kept unchanged.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, endpoints=np.array(eps), categories=cats, name_sim=name_sim,
             func_sim=func_sim, same_category=same_cat, func_bin_cuts=func_cuts,
             thresholds=json.dumps(thresholds), embedding_model=BGE)
    print(f"[fintrace similarity] {n} endpoints -> {out_path}")


# ---------------------------------------------------------------- correct calls and wrong calls
MAX_PER_TURN, MAX_PER_TRAJ, N_PREV = 3, 10, 5
NEAR_DUP_FUNC = 0.98
DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
DATE_KEYS = ("from", "to", "date")


def rng_for(*key):
    return random.Random(f"{SEED}|" + "|".join(map(str, key)))


def dumps(o):
    return json.dumps(o, ensure_ascii=False)


def error_reason(result):
    r = (result or "").strip()
    if r in ("", "[]", "{}", "null"):
        return f"empty:{r or '<blank>'}"
    if r[0] not in "[{":
        return "non_json:" + r[:60]
    if r.startswith("{"):
        try:
            if "Error Message" in json.loads(r):
                return "error_message"
        except json.JSONDecodeError:
            pass
    return None


def format_call(endpoint, args):
    return f"{endpoint}({dumps(args)})"


def build_state(query, prev_calls):
    lines = "\n".join(f"- {format_call(c['endpoint'], c['args'])}" for c in prev_calls[-N_PREV:]) or "none"
    return f"Question: {query}\nPrevious calls (most recent {N_PREV}):\n{lines}"


def action_c1(endpoint, args):
    return f"call {format_call(endpoint, args)}"


def action_c2(endpoint, args, description):
    return f"call {format_call(endpoint, args)} — {description}"


def shift_date(s, years):
    y, m, d = map(int, DATE_RE.match(s).groups())
    y += years
    if m == 2 and d == 29:
        d = 28
    return dt.date(y, m, d).isoformat()


def p4_options(args, data_source):
    opts = {}
    if data_source != "finmcp":
        sym = args.get("symbol")
        syms = args.get("symbols")
        if isinstance(sym, str) and sym.strip():
            opts["ticker"] = "symbol"
        elif isinstance(syms, list) and any(isinstance(s, str) and s.strip() for s in syms):
            opts["ticker"] = "symbols"
    date_keys = [k for k in DATE_KEYS if isinstance(args.get(k), str) and DATE_RE.match(args[k])]
    year_ok = isinstance(args.get("year"), int) and 1900 <= args["year"] <= 2100
    if date_keys or year_ok:
        opts["date"] = date_keys + (["year"] if year_ok else [])
    if args.get("period") in ("annual", "quarter"):
        opts["period"] = "period"
    return opts


def make_p4(args, data_source, query_tickers, ticker_pool, rng):
    """One argument error: a ticker from another query, all dates (and year) shifted by +-1..3 years, or the other
    period. Returns (subtype, new args, change) or None when the call has no such argument."""
    opts = p4_options(args, data_source)
    if not opts:
        return None
    sub = rng.choice(sorted(opts))
    new = json.loads(dumps(args))
    if sub == "ticker":
        pool = [t for t in ticker_pool if t not in query_tickers]
        repl = rng.choice(pool)
        if opts["ticker"] == "symbol":
            change = {"symbol": [args["symbol"], repl]}
            new["symbol"] = repl
        else:
            idx = [i for i, s in enumerate(args["symbols"]) if isinstance(s, str) and s.strip()]
            i = rng.choice(idx)
            change = {f"symbols[{i}]": [args["symbols"][i], repl]}
            new["symbols"][i] = repl
    elif sub == "date":
        years = rng.choice([-3, -2, -1, 1, 2, 3])
        change = {}
        for k in opts["date"]:
            new[k] = args[k] + years if k == "year" else shift_date(args[k], years)
            change[k] = [args[k], new[k]]
        change["_years"] = years
    else:
        new["period"] = "quarter" if args["period"] == "annual" else "annual"
        change = {"period": [args["period"], new["period"]]}
    return sub, new, change


def build_samples(testset_path, tools_path, sim_path, out_dir):
    """Cleaning funnel, correct calls (positives) and wrong calls (P2_bin1..5, P4) -> positives.jsonl, negatives.jsonl.

    Funnel: queries without golden calls dropped; calls with empty / non-JSON / "Error Message" results dropped;
    exact duplicate (endpoint, args) keep the first; endpoint must be in the catalog; per turn dedupe by endpoint and
    keep at most 3 (random); at most 10 per trajectory (random). State = question + the last 5 raw golden calls of
    earlier turns (arguments only)."""
    import numpy as np
    d = load_testset(testset_path)
    tools = {t["endpoint"]: t for t in read_jsonl(tools_path)}
    z = np.load(sim_path)
    eps = [str(e) for e in z["endpoints"]]
    eidx = {e: i for i, e in enumerate(eps)}
    name_sim, func_sim, same_cat = z["name_sim"], z["func_sim"], z["same_category"]
    cuts = z["func_bin_cuts"]

    # ticker pool per query (symbol / symbols values in golden args)
    q_tickers = {}
    for x in d:
        ts = set()
        for c in golden_calls(x):
            v = c["args"].get("symbol")
            if isinstance(v, str) and v.strip():
                ts.add(v.strip())
            for s in c["args"].get("symbols") or []:
                if isinstance(s, str) and s.strip():
                    ts.add(s.strip())
        q_tickers[x["id"]] = ts

    positives, calls_by_query = [], {}
    for x in d:
        calls = golden_calls(x)
        if calls:
            calls_by_query[x["id"]] = (x, calls)
    stage = {qid: list(cs) for qid, (x, cs) in calls_by_query.items()}
    for qid, cs in stage.items():  # error or empty results
        stage[qid] = [c for c in cs if not error_reason(c["result"])]
    for qid, cs in stage.items():  # exact duplicates
        seen, keep = set(), []
        for c in cs:
            k = (c["endpoint"], dumps(c["args"]))
            if k in seen:
                continue
            seen.add(k)
            keep.append(c)
        stage[qid] = keep
    for qid, cs in stage.items():  # endpoint in the catalog
        stage[qid] = [c for c in cs if c["endpoint"] in tools]
    for qid, cs in stage.items():  # per turn: dedupe by endpoint, keep at most 3
        keep = []
        by_turn = C.defaultdict(list)
        for c in cs:
            by_turn[c["turn"]].append(c)
        for turn, tcs in by_turn.items():
            seen, dedup = set(), []
            for c in tcs:
                if c["endpoint"] in seen:
                    continue
                seen.add(c["endpoint"])
                dedup.append(c)
            if len(dedup) > MAX_PER_TURN:
                chosen = rng_for("turn", qid, turn).sample(dedup, MAX_PER_TURN)
                dedup = [c for c in dedup if c in chosen]
            keep += dedup
        stage[qid] = sorted(keep, key=lambda c: c["call_idx"])
    for qid, cs in stage.items():  # at most 10 per trajectory
        if len(cs) > MAX_PER_TRAJ:
            chosen = rng_for("traj", qid).sample(cs, MAX_PER_TRAJ)
            stage[qid] = sorted(chosen, key=lambda c: c["call_idx"])

    for qid, cs in stage.items():
        x, raw_calls = calls_by_query[qid]
        for c in cs:
            prev = [p for p in raw_calls if p["turn"] < c["turn"]]
            t = tools[c["endpoint"]]
            positives.append({
                "pos_id": f"{qid}-{c['call_idx']}", "query_id": qid, "data_source": x["data_source"],
                "resource": x["resource"], "task_type_bucket": x["task_type_bucket"], "difficulty_tier": x["difficulty_tier"],
                "turn": c["turn"], "call_idx": c["call_idx"], "category": c["category"], "endpoint": c["endpoint"],
                "args": c["args"], "n_prev": len(prev), "is_first_call": not prev,
                "endpoint_in_history": c["endpoint"] in {q["endpoint"] for q in prev[-N_PREV:]},
                "desc_source": t["desc_source"],
                "state": build_state(x["source_query"], prev),
                "action_c1": action_c1(c["endpoint"], c["args"]),
                "action_c2": action_c2(c["endpoint"], c["args"], t["description"]),
                "result_head": c["result"][:300],
            })
    write_jsonl(out_dir / "positives.jsonl", positives)

    ticker_pool = sorted(set().union(*q_tickers.values()))
    negatives, missing = [], C.Counter()

    def neg(p, ptype, shown, desc_ep, args, **extra):
        # similarity between the correct endpoint and the substitute (P4: itself)
        i, j = eidx[p["endpoint"]], eidx[shown]
        n = {"neg_id": f"{p['pos_id']}-{ptype}", "pos_id": p["pos_id"], "query_id": p["query_id"],
             "perturb_type": ptype, "endpoint_shown": shown, "desc_endpoint": desc_ep, "args": args,
             "name_sim": float(name_sim[i, j]), "func_sim": float(func_sim[i, j]),
             "same_category": bool(same_cat[i, j]), "desc_source": tools[desc_ep]["desc_source"],
             "action_c1": action_c1(shown, args),
             "action_c2": action_c2(shown, args, tools[desc_ep]["description"])}
        n.update(extra)
        negatives.append(n)

    for p in positives:
        qid, src = p["query_id"], p["endpoint"]
        i = eidx[src]
        in_traj = {c["endpoint"] for c in calls_by_query[qid][1]}
        cand = np.array([j for j, e in enumerate(eps)
                         if e != src and e not in in_traj and func_sim[i, j] < NEAR_DUP_FUNC])
        fs = func_sim[i, cand]

        def pick(mask, ptype):
            pool = cand[mask]
            if len(pool) == 0:
                missing[ptype] += 1
                return None
            return eps[rng_for(ptype, p["pos_id"]).choice(sorted(pool.tolist()))]

        bins = np.digitize(fs, cuts)
        for b in range(5):  # P2: endpoint substitution, one per func_sim bin
            e = pick(bins == b, f"P2_bin{b + 1}")
            if e:
                neg(p, f"P2_bin{b + 1}", e, e, p["args"], sim_bin=b + 1)
        r = make_p4(p["args"], p["data_source"], q_tickers[qid], ticker_pool, rng_for("P4", p["pos_id"]))
        if r:
            sub, new_args, change = r
            neg(p, "P4", src, src, new_args, p4_subtype=sub, p4_change=change)
        else:
            missing["P4"] += 1
    write_jsonl(out_dir / "negatives.jsonl", negatives)
    print(f"[fintrace samples] positives {len(positives)}, negatives {len(negatives)}, without a draw {dict(missing)}")


# ---------------------------------------------------------------- per-argument inputs
def build_items(processed_dir, tools_path, out_dir):
    """Argument-check / drop-variant inputs for the positives and their P2 / P4 negatives in the C2 main set (both
    descriptions from the FMP docs)."""
    pos = {p["pos_id"]: p for p in read_jsonl(processed_dir / "positives.jsonl")}
    tools = {t["endpoint"].lower(): t for t in read_jsonl(tools_path)}
    allneg = [n for n in read_jsonl(processed_dir / "negatives.jsonl")
              if n["perturb_type"].startswith("P2_bin") or n["perturb_type"] == "P4"]
    neg = [n for n in allneg if pos[n["pos_id"]]["desc_source"] == "fmp_docs" and n["desc_source"] == "fmp_docs"]

    def info_for(endpoint):
        t = tools.get(endpoint.lower())
        params = {q["name"]: q for q in (t or {}).get("params") or []}

        def info_of(k):
            q = params.get(k)
            if q is None:
                return NOT_PARAM
            return f"type: {q.get('type')}; required: {str(q.get('required')).lower()}; example: {q.get('example')}"
        return info_of
    items, calls = [], []
    done = set()
    for n in neg:
        p = pos[n["pos_id"]]
        for role, x, ep in (("pos", p, p["endpoint"]), ("neg", n, n["endpoint_shown"])):
            cid = p["pos_id"] if role == "pos" else n["neg_id"]
            if cid in done:
                continue
            done.add(cid)
            name, rawa, args = split_call(x["action_c1"])
            fdesc = x["action_c2"][len(x["action_c1"]) + 3:]
            assert x["action_c2"] == x["action_c1"] + " — " + fdesc
            its, pids, dids, nid = call_items(cid, p["state"], name, fdesc, args, info_for(ep), rawa)
            items += its
            calls.append({"call_id": cid, "role": role, "pos_id": p["pos_id"], "neg_id": n["neg_id"] if role == "neg" else None,
                          "group": ("P2" if n["perturb_type"].startswith("P2") else "P4") if role == "neg" else None,
                          "perturb_type": n["perturb_type"] if role == "neg" else None, "query_id": p["query_id"],
                          "n_params": len(args), "param_items": pids, "drop_items": dids, "param_not_in_def": nid})
    write_items(out_dir, items, calls)
    print(f"[fintrace items] calls {len(calls)}, inputs {len(items)}")


def build_items_nodesc(items_dir, out_dir):
    n = reword_not_param(items_dir / "items.jsonl", out_dir / "items.jsonl")
    print(f"[fintrace items, re-worded] {n}")
