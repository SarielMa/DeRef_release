"""Jev submission by distinct input text, the full-text resubmission ladder, and the final per-item score files.

Work directory layout (`--work`):
  batch.json             the batches: [{"items", "out", "dataset", "kind"}] (kind: plain | plain_full | deref_full)
  r0.jsonl, r0_map.jsonl, r0.tsv, r0_counts.json     round 0: one row per distinct input not yet in the cache
  rR.jsonl, rR_map.jsonl, rR.tsv, rR_counts.json     round R >= 1: refused inputs, one more earliest return dropped
  out_rR/jev.jsonl, out_rR/jev_failed.jsonl          written by scripts/45_score_jev.py --list <work>/rR.tsv
  k.json                 extra ladder steps per item ("<items path>|<item_id>" -> k)
  jev_in/batch<j>.jsonl  the inputs Jev actually read (written by `final`)
  final_summary.json     checks of the final step

Steps:
  dedup   every distinct (context, question) over all batches gets one row; rows whose answer is already in the
          cache (pinned model) are not resubmitted. Map rows {"file", "out", "item_id", "uid", "key"}.
  next R  items of round R-1 refused with "max_tokens_exceeded" go one more step down the full-text deletion ladder
          (the same renderer as the local judges' full-history input, with the k+1 earliest steps' returns replaced by
          "  -> [return dropped]"; a DeRef + full-history input keeps its reference block as part of the tail) and are
          resubmitted; other errors are resubmitted unchanged. Exit 10: nothing left to send; exit 5: an item is still
          refused with all returns dropped (stop and report).
  final   every item takes the cached answer of the input it was sent with (k = 0: its own input, byte for byte; k >= 1:
          the ladder input) -> `out` rows {"item_id", "score", "model", "k", "input_key"}.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from deref.common import read_jsonl, write_jsonl
from deref.score import jev_client as J

REFUSED = "max_tokens_exceeded"
# Header of the reference block appended after the history by the DeRef inputs (kept byte-exact).
REF_HEAD = "\nExpected value (written before seeing the proposed value):\n"


def read_batches(path):
    """Batch list TSV: `<items.jsonl><TAB><output score file>[<TAB><dataset><TAB><kind>]` per line."""
    out = []
    for l in open(path):
        if not l.strip():
            continue
        c = l.rstrip("\n").split("\t")
        out.append({"items": c[0], "out": c[1], "dataset": c[2] if len(c) > 2 else None,
                    "kind": c[3] if len(c) > 3 else "plain"})
    return out


def round_paths(work, R):
    work = Path(work)
    return work / f"r{R}.jsonl", work / f"r{R}_map.jsonl", work / f"out_r{R}"


def write_round(work, R, bill, mp, info):
    sent, mfile, out = round_paths(work, R)
    write_jsonl(sent, bill)
    write_jsonl(mfile, mp)
    (Path(work) / f"r{R}.tsv").write_text(f"{sent}\t{out}\n")
    json.dump(info, open(Path(work) / f"r{R}_counts.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(info, ensure_ascii=False))


# ------------------------------------------------------------------------------------------------ dedup (round 0)
def cmd_dedup(batch_list, work, cache):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    batches = read_batches(batch_list)
    json.dump(batches, open(work / "batch.json", "w"), indent=1)
    db = J.snapshot(cache)
    uniq, mp, fs = {}, [], {}
    for b in batches:
        i, o = b["items"], b["out"]
        rows = read_jsonl(i)
        n0 = len(uniq)
        for r in rows:
            k = J.item_key(r)
            if k not in uniq:
                uniq[k] = {"item_id": f"u{len(uniq):05d}", "context": r["context"], "question": r["question"]}
            mp.append({"file": i, "out": o, "item_id": r["item_id"], "uid": uniq[k]["item_id"], "key": k})
        fs[i] = {"items": len(rows), "new distinct inputs": len(uniq) - n0}
    cached, bill = 0, []
    for k, u in uniq.items():
        v = db.execute("SELECT v FROM ans WHERE k=?", (k,)).fetchone()
        if v is not None:
            assert json.loads(v[0]).get("model") == J.PINNED, "cached answer from another model version"
            cached += 1
        else:
            bill.append(u)
    write_round(work, 0, bill, mp, {"round": 0, "files": fs, "items": len(mp), "distinct inputs": len(uniq),
                                    "in cache (pinned, not billed)": cached, "to bill": len(bill)})


# ------------------------------------------------------------------------------------------------ ladder
def full_text_module():
    """The full-history renderer of the judge inputs (deref.inputs.full_text). Required interface:
      render(question, pieces, drop_ret=0, drop_steps=0) -> str   full-history state text
      step_of(label) -> int                                        step number of a piece label
      ladder_parts(data_dir, dataset) -> {item_id: (question, pieces, plain_tail)}
        pieces = [(label, text)] of the history (calls and returns, oldest first, without the question);
        plain_tail = the text that follows the history in the Plain input of that argument item.
    """
    try:
        from deref.inputs import full_text
    except ImportError as e:
        raise NotImplementedError("deref.inputs.full_text is required for the ladder") from e
    missing = [f for f in ("render", "step_of", "ladder_parts") if not hasattr(full_text, f)]
    if missing:
        raise NotImplementedError(f"deref.inputs.full_text lacks {missing}")
    return full_text


class Ladder:
    """Rebuild a full-history item at ladder level (dr0 + k, 0) with the same render / pieces / tail as its local-judge
    input. `parts(file, row) -> (question, pieces, plain_tail)`."""

    def __init__(self, parts, render, step_of, kinds):
        self.parts, self.render, self.step_of, self.kinds = parts, render, step_of, kinds
        self.memo = {}

    def level0(self, f, x):
        """The local-judge input = render(question, pieces, dr0, ds0) + tail, byte for byte.
        Returns (question, pieces, tail, (dr0, ds0), n steps)."""
        key = (f, x["item_id"])
        if key in self.memo:
            return self.memo[key]
        q, pieces, tail0 = self.parts(f, x)
        n = len({self.step_of(l) for l, _ in pieces})
        levels = [(0, 0)] + [(dr, 0) for dr in range(1, n + 1)] + [(n, ds) for ds in range(1, n + 1)]
        hit = []
        for lv in levels:
            r = self.render(q, pieces, *lv)
            if x["context"].startswith(r) and x["context"].endswith(tail0):
                tail = x["context"][len(r):]
                if self.kinds[f] == "deref_full":  # tail = reference block + Plain tail, block untouched
                    ok = tail.startswith(REF_HEAD) and tail.count(REF_HEAD) == 1 and tail.endswith(tail0)
                else:
                    ok = tail == tail0
                if ok:
                    hit.append((lv, tail))
        assert len(hit) == 1, (f, x["item_id"], [h[0] for h in hit])
        lv, tail = hit[0]
        if "level" in x:
            assert tuple(x["level"]) == lv, (f, x["item_id"])
        assert lv == (0, 0) or x.get("cut", True), (f, x["item_id"])
        self.memo[key] = (q, pieces, tail, lv, n)
        return self.memo[key]

    def build(self, f, x, k):
        """Jev input row for item x with k more earliest returns dropped; None if all returns are dropped already."""
        if k == 0:
            return dict(x)
        q, pieces, tail, (dr0, ds0), n = self.level0(f, x)
        if ds0 > 0 or dr0 + k > n:
            return None
        ctx = self.render(q, pieces, dr0 + k, 0) + tail
        assert ctx != x["context"]
        return {**x, "context": ctx}


def make_ladder(batches, data_dir, parts=None, render=None, step_of=None):
    """Ladder over the batches; by default the parts and renderer come from deref.inputs.full_text."""
    kinds = {b["items"]: b["kind"] for b in batches}
    if parts is None:
        ft = full_text_module()
        tables = {}

        def parts(f, x):
            d = next(b["dataset"] for b in batches if b["items"] == f)
            if d not in tables:
                tables[d] = ft.ladder_parts(data_dir, d)
            return tables[d][x["item_id"]]
        render, step_of = ft.render, ft.step_of
    return Ladder(parts, render, step_of, kinds)


def load_k(work):
    p = Path(work) / "k.json"
    return json.load(open(p)) if p.exists() else {}


def cmd_next(work, cache, R, data_dir, ladder=None):
    work = Path(work)
    batches = json.load(open(work / "batch.json"))
    sent, mfile, out = round_paths(work, R - 1)
    ok = {r["item_id"]: r for r in read_jsonl(out / "jev.jsonl")}
    bad = {r["item_id"]: r["error"] for r in read_jsonl(out / "jev_failed.jsonl")}
    ids = {r["item_id"] for r in read_jsonl(sent)}
    assert set(ok) | set(bad) == ids and not set(ok) & set(bad), f"round {R - 1} incomplete: rerun it first"
    assert {r["model"] for r in ok.values()} <= {J.PINNED}, "a reply from another model version: stop"
    budget = [u for u, e in bad.items() if e.startswith("Budget")]
    assert not budget, f"round {R - 1}: budget stop ({len(budget)})"
    K = load_k(work)
    rows = {b["items"]: {r["item_id"]: r for r in read_jsonl(b["items"])} for b in batches}
    L = ladder or make_ladder(batches, data_dir)
    todo, exhausted, st = [], [], Counter()
    for m in read_jsonl(mfile):
        e = bad.get(m["uid"])
        if e is None:
            continue
        kk = f"{m['file']}|{m['item_id']}"
        k0 = K.get(kk, 0)
        k = k0 + 1 if REFUSED in e else k0
        st[f"{m['file']}: {'refused -> k+1' if REFUSED in e else 'other error -> same k'}"] += 1
        x = L.build(m["file"], rows[m["file"]][m["item_id"]], k)
        if x is None:
            exhausted.append({"file": m["file"], "item_id": m["item_id"], "k": k0, "error": e})
            continue
        K[kk] = k
        todo.append((m, x, k))
    if exhausted:
        json.dump(exhausted, open(work / "exhausted.json", "w"), indent=1, ensure_ascii=False)
        print(f"[ladder] {len(exhausted)} items refused with all returns dropped: stop and report ({work / 'exhausted.json'})")
        sys.exit(5)
    json.dump(K, open(work / "k.json", "w"))
    if not todo:
        print(f"[ladder] round {R - 1} had no refusals: done")
        sys.exit(10)
    db = J.snapshot(cache)
    uniq, mp, cached = {}, [], 0
    for m, x, k in todo:
        h = J.item_key(x)
        if h not in uniq:
            v = db.execute("SELECT v FROM ans WHERE k=?", (h,)).fetchone()
            if v is not None:
                assert json.loads(v[0]).get("model") == J.PINNED
                cached += 1
            uniq[h] = None if v is not None else {"item_id": f"r{R}u{len(uniq):05d}", "context": x["context"],
                                                   "question": x["question"]}
        if uniq[h] is not None:
            mp.append({"file": m["file"], "out": m["out"], "item_id": m["item_id"], "uid": uniq[h]["item_id"], "key": h, "k": k})
    bill = [u for u in uniq.values() if u is not None]
    write_round(work, R, bill, mp, {"round": R, **dict(st), "items to resubmit": len(todo), "distinct inputs": len(uniq),
                                    "in cache": cached, "to bill": len(bill), "k max": max(k for _, _, k in todo)})
    if not bill:
        sys.exit(10)


# ------------------------------------------------------------------------------------------------ final
def cmd_final(work, cache, data_dir, ladder=None):
    work = Path(work)
    batches = json.load(open(work / "batch.json"))
    db = J.snapshot(cache)
    K = load_k(work)
    L = ladder if ladder is not None else (make_ladder(batches, data_dir) if K else None)
    st = {}
    rounds, R = [], 0
    while round_paths(work, R)[2].exists():
        rounds.append(R)
        R += 1
    models, billed_ok, refused, other = set(), 0, 0, 0
    replies0 = {}
    for R in rounds:
        out = round_paths(work, R)[2]
        ok = read_jsonl(out / "jev.jsonl")
        bad = read_jsonl(out / "jev_failed.jsonl")
        if R == 0:
            replies0 = {r["item_id"]: r["score"] for r in ok}
        models |= {r["model"] for r in ok}
        billed_ok += sum(not r["cached"] for r in ok)
        refused += sum(REFUSED in r["error"] for r in bad)
        other += sum(REFUSED not in r["error"] for r in bad)
    st["(a) replies, all rounds: models"] = sorted(models)
    assert set(models) <= {J.PINNED}, "a reply from another model version: stop"
    map0 = round_paths(work, 0)[1]
    uid0 = {(m["file"], m["item_id"]): m["uid"] for m in read_jsonl(map0)} if map0.exists() else {}
    ktab = []
    for j, b in enumerate(batches):
        i, o = b["items"], b["out"]
        rows = read_jsonl(i)
        jin, got, ks = [], [], []
        for x in rows:
            k = K.get(f"{i}|{x['item_id']}", 0)
            if k == 0:
                y = dict(x)
                lv = lvj = None
            else:
                y = L.build(i, x, k)
                q, pieces, tail, lv, n = L.level0(i, x)
                # (e) the Jev input is the same render with k more earliest returns dropped
                assert y["context"] == L.render(q, pieces, lv[0] + k, 0) + tail and y["context"] != x["context"]
                assert y["question"] == x["question"]
                lvj = [lv[0] + k, 0]
            jin.append({**y, "k": k, "level_local": list(lv) if lv else None, "level_jev": lvj})
            h = J.item_key(y)
            v = db.execute("SELECT v FROM ans WHERE k=?", (h,)).fetchone()
            assert v is not None, f"no Jev answer for {i} {x['item_id']} at k={k}: rerun the missing round"
            c = json.loads(v[0])
            assert c["model"] == J.PINNED
            p = float(c["answers"]["q"]["noul"])
            u = uid0.get((i, x["item_id"]))
            if k == 0 and u in replies0:
                assert abs(p - replies0[u]) < 1e-12
            got.append({"item_id": x["item_id"], "score": p, "model": c["model"], "k": k, "input_key": h})
            ks.append(k)
        write_jsonl(work / "jev_in" / f"batch{j}.jsonl", jin)
        ids = [r["item_id"] for r in rows]
        st[f"(c) {i}: items / out / duplicates / missing"] = [len(ids), len(got), len(got) - len({g['item_id'] for g in got}),
                                                            len(set(ids) - {g['item_id'] for g in got})]
        kk = np.array(ks)
        pos = kk[kk > 0]
        st[f"(e) {i}: items / identical to the local-judge input / k >= 1 / k min-median-max"] = [
            len(ks), int((kk == 0).sum()), int(len(pos)),
            [int(pos.min()), float(np.median(pos)), int(pos.max())] if len(pos) else None]
        ktab.append({"file": i, "items": len(ks), "identical": int((kk == 0).sum()), "k>=1": int(len(pos)),
                     **{f"k={v}": int((kk == v).sum()) for v in sorted(set(pos.tolist()))}})
        Path(o).parent.mkdir(parents=True, exist_ok=True)
        with open(o, "w") as f:
            f.writelines(json.dumps(g) + "\n" for g in got)
    for R in rounds:
        c = json.load(open(work / f"r{R}_counts.json"))
        st[f"(d) round {R}: distinct inputs / to bill"] = [c["distinct inputs"], c["to bill"]]
    st["(f) requests refused (max_tokens_exceeded) / other errors, all rounds"] = [refused, other]
    st["(f) replies billed (accepted, not cached), all rounds"] = billed_ok
    st["k table"] = ktab
    json.dump(st, open(work / "final_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(st, indent=1, ensure_ascii=False))
