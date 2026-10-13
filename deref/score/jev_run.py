"""Jev runner: one Noul request per input row (state = context, instructions = question).

Input: a list file, one line per batch: `<items.jsonl><TAB><output dir>` (paths relative to the working directory or
absolute). Output per batch: `<output dir>/jev.jsonl` rows {"item_id", "score", "model", "cached", "latency_ms"} and
`<output dir>/jev_failed.jsonl` rows {"item_id", "error"}.

Order of checks:
1. Local length check: every input's Qwen3-8B token count of `context + "\n" + question` (no special tokens) must be
   <= --probe-max, else the run stops before any request. Default 9,000 (the longest input of an early length probe
   that Jev read without truncation). The full-history inputs (Plain-full, DeRef + full history) were run with
   --probe-max 40000, i.e. the local check no longer blocks them and Jev's own limit decides: inputs Jev refuses come
   back as HTTP 400 "max_tokens_exceeded" in jev_failed.jsonl and go down the deletion ladder (jev_ladder.py).
2. Stop file: a run refuses to start while it exists (written when a budget rule stops a run).
3. Budgets: --max-cost (reported dollars, cumulative over runs via the state file), --max-calls (billed requests of
   this run), optional half-way rule (--half-cost; 0 = off).
Exit codes: 0 done; 2 a reply came from another model version; 3 some requests failed (rerun fills from the cache);
4 budget reached.
"""
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from deref.score import jev_client as J
from deref.score import models as M

PROBE_MAX = 9000
TOTAL_UNIQUE = 181888  # distinct inputs of the Plain-style run (used by the half-way rule)
HALF_COST = 7.0


def read_list(path):
    """[(items path, output dir)] from a TSV list file."""
    return [tuple(Path(p) for p in l.rstrip("\n").split("\t")) for l in open(path) if l.strip()]


def length_check(files, probe_max, tokenizer_path=None):
    """Number of inputs over `probe_max` Qwen3-8B tokens (context + "\n" + question, no special tokens)."""
    from transformers import AutoTokenizer
    src, rev = (tokenizer_path, None) if tokenizer_path else M.QWEN3_8B
    tok = AutoTokenizer.from_pretrained(src, revision=rev)
    over = 0
    for i, _, rows in files:
        n = []
        for b in range(0, len(rows), 2000):  # chunks keep memory bounded on very long files
            n += [len(x) for x in tok([r["context"] + "\n" + r["question"] for r in rows[b:b + 2000]],
                                      add_special_tokens=False)["input_ids"]]
        over += sum(x > probe_max for x in n)
        print(f"[length] {i}: {len(rows)} inputs, {len({(r['context'], r['question']) for r in rows})} "
              f"distinct, max {max(n)} Qwen tokens (probe limit {probe_max}), over {sum(x > probe_max for x in n)}")
    return over


def run(a):
    """Run all batches of the list file `a.list` (argparse namespace from scripts/45_score_jev.py)."""
    cache = Path(a.cache)
    state = Path(a.state) if a.state else cache.parent / "jev_run_state.json"
    stopf = Path(a.stop) if a.stop else cache.parent / "jev_STOP"
    files = [(i, o, [json.loads(l) for l in open(i)]) for i, o in read_list(a.list)]
    over = length_check(files, a.probe_max, a.tokenizer_path)
    if over:
        sys.exit("[length] inputs longer than the Jev probe: stop")
    if stopf.exists():
        sys.exit(f"[jev] {stopf} exists ({stopf.read_text().strip()}): stopped by a budget rule")
    print(f"[jev] state {state}, max cost ${a.max_cost}, half-way rule "
          + (f"${a.half_cost} at {a.total_unique // 2} billed" if a.half_cost else "off"))
    prev = json.loads(state.read_text()) if state.exists() else {"billed": 0, "cost": 0.0}
    print(f"[jev] earlier runs: billed {prev['billed']}, cost ${prev['cost']:.4f}")
    judge = J.make_judge(cache, a.max_calls)

    def tot():
        return prev["billed"] + judge.billed, prev["cost"] + judge.cost

    def save():
        b, c = tot()
        state.write_text(json.dumps({"billed": b, "cost": c}))

    def stop(msg):
        if not stopf.exists():
            stopf.write_text(msg + "\n")
        return msg
    lock = threading.Lock()
    prog = {"done": 0, "fail": 0, "next": 5000, "halfchecked": False}

    def ask(x):
        with lock:
            b, c = tot()
            if c >= a.max_cost:
                return x["item_id"], None, stop(f"Budget: max cost ${a.max_cost} reached")
            if a.half_cost and b >= a.total_unique // 2 and not prog["halfchecked"]:
                prog["halfchecked"] = True
                print(f"[half] billed {b} / {a.total_unique}, cost ${c:.4f} (rule: stop if > ${a.half_cost})", flush=True)
                if c > a.half_cost:
                    stop(f"Budget: half-way cost ${c:.2f} > ${a.half_cost}")
            if stopf.exists():
                return x["item_id"], None, "Budget: " + stopf.read_text().strip()
            if judge.billed >= prog["next"]:
                prog["next"] += 5000
                save()
                print(f"[progress] billed {b} / {a.total_unique}, cost ${c:.4f}, failures {prog['fail']}", flush=True)
        try:
            r = judge.ask(x["context"], J.questions_for(x["question"]))
            return x["item_id"], r, None
        except J.Budget as e:
            return x["item_id"], None, f"Budget: {e}"
        except Exception as e:  # noqa: BLE001  record and continue; a rerun fills from the cache
            with lock:
                prog["fail"] += 1
            return x["item_id"], None, f"{type(e).__name__}: {str(e)[:300]}"
    models, failed = set(), 0
    for i, o, rows in files:
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(ask, rows))
        ok = [(k, r) for k, r, e in res if r is not None]
        bad = [(k, e) for k, r, e in res if r is None]
        models |= {r.model for _, r in ok}
        failed += len(bad)
        o.mkdir(parents=True, exist_ok=True)
        with open(o / "jev.jsonl", "w") as f:
            f.writelines(json.dumps({"item_id": k, "score": float(r.answers["q"]["noul"]), "model": r.model,
                                     "cached": r.cached, "latency_ms": r.latency_ms}) + "\n" for k, r in ok)
        with open(o / "jev_failed.jsonl", "w") as f:
            f.writelines(json.dumps({"item_id": k, "error": e}) + "\n" for k, e in bad)
        save()
        st = judge.stats.summary()
        print(f"[jev] {i}: scored {len(ok)}, failed {len(bad)}; cumulative billed {tot()[0]}, cached "
              f"{st['cached']}, retries {st['retries']}, cost ${tot()[1]:.4f}; latency mean {st['latency_ms_mean']} ms",
              flush=True)
        if bad:
            print(f"[jev] first failure: {bad[0][1]}")
        if models - {J.PINNED}:
            print(f"[jev] STOPPED: answering version differs: {models}")
            sys.exit(2)
        if any(e.startswith("Budget") for _, e in bad):
            print("[jev] STOPPED: budget reached; rerun with a higher cap resumes from the cache")
            sys.exit(4)
    print(f"[jev] answering models: {models}")
    if a.repeat and not failed:
        allx = list({(r["context"], r["question"]): r for _, _, rows in files for r in rows}.values())
        pick = random.Random(f"{J.SEED}|repeat|v2").sample(allx, min(a.repeat, len(allx)))
        cache_db, judge.db = judge.db, None  # bypass the cache for the repeats only
        agree, diffs = 0, []
        for x in pick:
            q = J.questions_for(x["question"])
            p1 = json.loads(cache_db.execute("SELECT v FROM ans WHERE k=?",
                                             (J.cache_key(judge.name, judge.model, x["context"], q),))
                            .fetchone()[0])["answers"]["q"]["noul"]
            p2 = judge.ask(x["context"], q).answers["q"]["noul"]
            diffs.append(abs(p1 - p2))
            agree += (p1 >= 0.5) == (p2 >= 0.5)
        judge.db = cache_db
        print(f"[repeat] {len(pick)} requests re-asked: same side of 0.5 {agree}/{len(pick)}, exact same value "
              f"{sum(d == 0 for d in diffs)}/{len(pick)}, max |diff| {max(diffs):.4f}; this run's cost now ${judge.cost:.4f}")
    save()
    print(f"[jev] cumulative over runs: billed {tot()[0]}, cost ${tot()[1]:.4f}")
    judge.close()
    print("=== done" if not failed else f"=== {failed} failures: rerun to fill from the cache")
    sys.exit(3 if failed else 0)
