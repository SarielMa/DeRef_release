"""Shared pieces of every generation prompt: instruction texts, the full-history rendering, the 32,768-token deletion
ladder, chat rendering (thinking off / on) and the split of a Plain argument text into its parts.

Prompt layout (user message, thinking off):

    Question: <request>
    History (all earlier calls and their returns, oldest first):
    [call, step 1] name(args)
    [return, step 1] <return text>
    ...
    <tail>

`<tail>` is `<Function line>\\n<Parameter line>\\n\\n<instruction>` for the per-argument prompts (locator, derivation)
and a method-specific text for Twin / Revise. All texts are byte-exact; do not edit them.
"""
import json
import re

SYSTEM = "You are a careful evaluator of tool calls made by an AI assistant."

# Locator instruction (rule v2 wording: the base sentence plus the sentence that forbids quoting the descriptions).
# Every derivation / evidence-search prompt is built from the locator prompt of its argument.
ASK = ("Quote the exact text from the context above that the value of parameter {name} should be taken or derived from. "
       "Give at most 2 quotes, each on its own line. If there is no such text, answer NONE."
       " Only quote from the user question or from the previous calls and their results, not from the function or "
       "parameter descriptions.")

# Derivation instruction (DeRef family and SGV-style prior).
F2 = ("Before seeing any proposed value: what should the value of parameter {name} be in this call, and where in the "
      "context does it come from? Quote the source text if there is one, or say that the context does not determine it. "
      "Answer in at most 3 sentences.")

# Revise instructions.
RUBRIC = "Rubric: Each argument value should be correct for this call, given the question and the history."
REV = ("Revise the proposed call so that each argument value is correct, given the question and the history. Keep values that are "
       "already correct. Output only a JSON object that maps each argument name to its value.")

LOC_LIMIT, MIN_Q = 32768, 40  # prompt-token limit (Qwen3.8-27B, thinking-off chat text); shortest question after cutting
OFF_TAIL, ON_TAIL_27B, ON_TAIL_8B = "<think>\n\n</think>\n\n", "<think>\n", ""
HISTORY_HEADER = "History (all earlier calls and their returns, oldest first):\n"


def step_of(lab):
    return int(lab.rsplit("step ", 1)[1])


# ---------------------------------------------------------------- rendering and the 32,768-token ladder
def render_tail(s, tail, drop_ret=0, drop_steps=0, q=None):
    """User message: question, full history (minus the returns of the `drop_ret` earliest steps and minus the
    `drop_steps` earliest steps), then `tail`. Removed lines are not marked."""
    steps = sorted({step_of(l) for l, _ in s["pieces"]})
    gone_ret, gone = set(steps[:drop_ret]), set(steps[:drop_steps])
    h = [f"[{lab}] {t}" for lab, t in s["pieces"]
         if step_of(lab) not in gone and not (lab.startswith("return") and step_of(lab) in gone_ret)]
    return (f"Question: {s['question'] if q is None else q}\n" + HISTORY_HEADER
            + ("\n".join(h) if h else "none") + f"\n{tail}")


def ladder_tail(s, tail, tok):
    """Deletion levels in order: none -> drop the returns of the 1, 2, ... earliest steps -> drop all returns and the
    1, 2, ... earliest whole steps -> cut the question in the middle (first / last halves joined by " … ", 8 tokens
    per step, at least MIN_Q tokens). Yields (level label, user message)."""
    n = len({step_of(l) for l, _ in s["pieces"]})
    yield "none", render_tail(s, tail)
    for dr in range(1, n + 1):
        yield f"returns dropped for {dr} earliest step(s)", render_tail(s, tail, drop_ret=dr)
    for dsp in range(1, n + 1):
        yield f"all returns and {dsp} earliest step(s) dropped", render_tail(s, tail, drop_ret=n, drop_steps=dsp)
    ids = tok(s["question"], add_special_tokens=False)["input_ids"]
    for keep in range(len(ids) - 8, MIN_Q - 1, -8):
        q2 = tok.decode(ids[:keep // 2]) + " … " + tok.decode(ids[len(ids) - (keep - keep // 2):])
        yield f"question cut to {keep} tokens", render_tail(s, tail, drop_ret=n, drop_steps=n, q=q2)


def chat(tok, user, system=SYSTEM):
    """Chat text with thinking off (the template adds an empty think block)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def ntok(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


def fit(s, tail, tok):
    """First ladder level whose Qwen3.8-27B thinking-off chat text has <= LOC_LIMIT tokens.
    -> (level label, user message, chat text, tokens)."""
    for lab, user in ladder_tail(s, tail, tok):
        text = chat(tok, user)
        n = ntok(tok, text)
        if n <= LOC_LIMIT:
            return lab, user, text, n
    raise SystemExit("prompt still over the limit")


# ---------------------------------------------------------------- per-argument pieces
def split_main(ctx):
    """Plain argument text -> (state part, Function line, Parameter line)."""
    head, _ = ctx.rsplit("\nProposed value for ", 1)
    i = head.rindex("\nFunction: ")
    fp = head[i + 1:]
    j = fp.index("\nParameter: ")
    return head[:i], fp[:j], fp[j + 1:]


def without_value(ctx):
    """Plain argument text without its `Proposed value for ...` line."""
    head, _ = ctx.rsplit("\nProposed value for ", 1)
    return head


def pname(item_id):
    """Parameter name from an argument id (`<call>|p|<param>` or `<call>|p|<j>:<param>`)."""
    return item_id.rsplit("|p|", 1)[1].split(":", 1)[-1]


def locator_tail(fl, pl, name):
    return f"{fl}\n{pl}\n\n{ASK.format(name=name)}"


def vline(p, v):
    """Value line inserted after the Parameter line (evidence search, DeRef-v)."""
    return f"Proposed value for {p}: {json.dumps(v, ensure_ascii=False)}"


ASK_PATTERN = re.compile(re.escape(ASK.split("{name}")[0]) + r"(.+?)" + re.escape(ASK.split("{name}")[1]))


def derivation_text(locator_text):
    """Derivation prompt = locator chat text with the locator instruction replaced by F2 (same parameter name)."""
    m = ASK_PATTERN.search(locator_text)
    assert m and len(ASK_PATTERN.findall(locator_text)) == 1
    return locator_text[:m.start()] + F2.format(name=m.group(1)) + locator_text[m.end():]


def insert_value_line(text, pl, param, value):
    """Insert the value line after the (unique) Parameter line."""
    anchor = "\n" + pl + "\n\n"
    assert text.count(anchor) == 1
    return text.replace(anchor, "\n" + pl + "\n" + vline(param, value) + "\n\n")


# ---------------------------------------------------------------- chat texts
def split_chat(t):
    """Thinking-off Qwen chat text -> (system, user)."""
    a, b = "<|im_start|>system\n", "<|im_end|>\n<|im_start|>user\n"
    c = "<|im_end|>\n<|im_start|>assistant\n" + OFF_TAIL
    assert t.startswith(a) and t.count(b) == 1 and t.endswith(c), t[:80]
    i = t.index(b)
    return t[len(a):i], t[i + len(b):-len(c)]


def thinking_on(tok, off_text, small_model=False):
    """Thinking-on re-render of a thinking-off chat text (same system / user messages).

    Qwen3.8-27B: `enable_thinking=True, reasoning_effort="medium"` (for "medium" the template adds nothing to the system
    message); the only difference to the thinking-off text is the tail `<think>\\n\\n</think>\\n\\n` -> `<think>\\n`.
    Qwen3-8B: `enable_thinking=True` (reasoning_effort has no effect, asserted); the empty think block is dropped."""
    sysm, user = split_chat(off_text)
    msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": user}]
    off = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert off == off_text
    if small_model:
        on = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        on_m = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                       reasoning_effort="medium")
        assert on_m == on
        tail = ON_TAIL_8B
    else:
        on = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                     reasoning_effort="medium")
        tail = ON_TAIL_27B
    assert off.endswith(OFF_TAIL) and on == off[:-len(OFF_TAIL)] + tail
    return on
