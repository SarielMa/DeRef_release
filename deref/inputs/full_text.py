"""Full-history judge inputs (Plain-full, DeRef + full history) and the 32,768-token deletion ladder.

Render: "Question: <request>\\nPrevious calls (all earlier calls, oldest first):\\n" + every earlier call
(`- name(args)`) with its complete return (`  -> <return>`), or "none". Returns are never cut by length.
Ladder levels (drop_returns, drop_steps): (0, 0); (1, 0) … (n, 0) = the returns of the 1 … n earliest steps replaced by
"  -> [return dropped]"; (n, 1) … (n, n) = all returns dropped and the 1 … n earliest steps removed. The question is never
cut. The chosen level is the first whose full Qwen judge prompt (system prompt, user "<context>\\n\\n<question> Answer Yes
or No.", thinking off) has at most 32,768 tokens under both the Qwen3-8B and the Qwen3.8-27B tokenizer (an approximate
count, state + tail + template overhead, skips levels far over the limit; the exact prompt is counted near the limit).
No level fitting is an error (no item is dropped).
Plain-full = full render + Plain argument tail (ComplexFuncBench, FinTrace; FC-RewardBench has no history, its
Plain-full input is the Plain input). DeRef + full history = the same with the DeRef block (HEAD + reference) between
the history and the tail: ComplexFuncBench pastes the block into the Plain-full input without re-running the ladder
(error if over the limit); FinTrace re-runs the ladder with the block as part of the tail.
`deeper_input` gives the input with k more returns dropped (used when a judge API refuses an input as too long)."""
from deref.inputs import state as ST
from deref.inputs.plain import base_parts

LIMIT = 32768
HEADER = "Previous calls (all earlier calls, oldest first):"
DROPPED = "  -> [return dropped]"


def step_of(lab):
    return int(lab.rsplit("step ", 1)[1])


def render(question, pieces, drop_ret=0, drop_steps=0):
    """pieces = [(label, text)] without the question; earliest `drop_ret` steps lose their returns, earliest
    `drop_steps` steps are removed."""
    steps = sorted({step_of(l) for l, _ in pieces})
    gone_ret, gone = set(steps[:drop_ret]), set(steps[:drop_steps])
    lines = []
    for lab, t in pieces:
        s = step_of(lab)
        if s in gone:
            continue
        if lab.startswith("call"):
            lines.append(f"- {t}")
        else:
            lines.append(DROPPED if s in gone_ret else f"  -> {t}")
    return f"Question: {question}\n{HEADER}\n" + ("\n".join(lines) if lines else "none")


def qwen_prompt(tok, ctx, q):
    """Qwen judge prompt (thinking off) of a judge input."""
    msgs = [{"role": "system", "content": ST.SYSTEM}, {"role": "user", "content": f"{ctx}\n\n{q} Answer Yes or No."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def history_parts(rec):
    """(question, pieces without the question) of a record."""
    return rec["request"], ST.pieces(rec)[1:]


def levels_of(pieces):
    n = len({step_of(l) for l, _ in pieces})
    return n, [(0, 0)] + [(dr, 0) for dr in range(1, n + 1)] + [(n, ds_) for ds_ in range(1, n + 1)]


class Builder:
    """32,768-token ladder under two tokenizers (Qwen3-8B, Qwen3.8-27B), with a per-state cache of rendered states."""

    def __init__(self, tok8, tok27):
        self.toks = (tok8, tok27)
        self.overhead = [self.ntok(tk, qwen_prompt(tk, "", "")) for tk in self.toks]  # template + "Answer Yes or No."
        self.cache = {}

    @staticmethod
    def ntok(tok, t):
        return len(tok(t, add_special_tokens=False)["input_ids"])

    def state_len(self, key, question, pieces, level):
        """token counts (8B, 27B) of the rendered state at ladder level (drop_ret, drop_steps)."""
        k = (key, level)
        if k not in self.cache:
            txt = render(question, pieces, *level)
            self.cache[k] = (txt, [self.ntok(tk, txt) for tk in self.toks])
        return self.cache[k]

    def build(self, key, question, pieces, tail, q):
        """(context, tokens, level): smallest ladder level that fits; exact count of the final prompt near the limit."""
        _, levels = levels_of(pieces)
        tl = [self.ntok(tk, tail + "\n\n" + q) for tk in self.toks]
        for lv in levels:
            txt, sl = self.state_len(key, question, pieces, lv)
            approx = max(a + b + c for a, b, c in zip(sl, tl, self.overhead))
            if approx > LIMIT + 64:
                continue
            ctx = txt + tail
            exact = max(self.ntok(tk, qwen_prompt(tk, ctx, q)) for tk in self.toks) if approx > LIMIT - 256 else approx
            if exact <= LIMIT:
                return ctx, exact, lv
        raise RuntimeError(f"no ladder level fits: {key}")

    def exact(self, ctx, q):
        return max(self.ntok(tk, qwen_prompt(tk, ctx, q)) for tk in self.toks)


def plain_full_item(builder, rec, arg):
    _, _, tail, q = base_parts(rec, arg)
    question, pcs = history_parts(rec)
    ctx, n, lv = builder.build(rec["state_id"], question, pcs, tail, q)
    return {"item_id": arg["item_id"], "context": ctx, "question": q, "tokens": n, "cut": lv != (0, 0), "level": list(lv),
            "state": rec["state_id"]}


def build_plain_full(records, builder, analysis_only=True):
    """Plain-full rows (ComplexFuncBench, FinTrace; FC-RewardBench has no history and yields none). A state without
    earlier calls (first step) renders "none" as its history."""
    return [plain_full_item(builder, r, a) for r, a in ST.arguments(records, analysis_only) if r["dataset"] != "fcrb"]


def deref_full_item(builder, rec, arg, block):
    """DeRef + full history input; `block` = HEAD + reference text (see reference.HEAD)."""
    _, _, tail, q = base_parts(rec, arg)
    question, pcs = history_parts(rec)
    ctx0, n0, lv0 = builder.build(rec["state_id"], question, pcs, tail, q)  # the Plain-full input
    if rec["dataset"] == "cfb":  # no ladder: block pasted into the Plain-full input
        ctx = ctx0[:len(ctx0) - len(tail)] + block + tail
        n = builder.exact(ctx, q)
        if n > LIMIT:
            raise RuntimeError(f"{arg['item_id']}: {n} tokens > limit (no ladder for this dataset)")
        lv = lv0
    else:
        ctx, n, lv = builder.build(rec["state_id"], question, pcs, block + tail, q)
    return {"item_id": arg["item_id"], "context": ctx, "question": q, "tokens": n, "cut": lv != (0, 0), "level": list(lv),
            "level_plain_full": list(lv0), "state": rec["state_id"]}


def deeper_input(rec, row, k):
    """The full-history input `row` (Plain-full or DeRef + full, carrying its `level`) with k more of the earliest
    returns dropped (level (dr + k, 0)); everything after the history (block, tail) unchanged. k = 0 returns the row;
    None when the input already drops all returns or steps."""
    if k == 0:
        return dict(row)
    question, pcs = history_parts(rec)
    n, _ = levels_of(pcs)
    dr0, ds0 = row["level"]
    head = render(question, pcs, dr0, ds0)
    assert row["context"].startswith(head), row["item_id"]
    tail = row["context"][len(head):]
    if ds0 > 0 or dr0 + k > n:
        return None
    ctx = render(question, pcs, dr0 + k, 0) + tail
    assert ctx != row["context"]
    return {**row, "context": ctx, "k": k}


def ladder_parts(data_dir, dataset):
    """{argument item_id: (question, pieces without the question, Plain tail)} for every argument of a dataset in the
    data package (the parts the resubmission ladder re-renders; see `deeper_input`)."""
    out = {}
    for rec in ST.load_records(data_dir, dataset):
        question, pcs = history_parts(rec)
        for arg in rec["arguments"]:
            out[arg["item_id"]] = (question, pcs, ST.argument_tail(rec, arg))
    return out
