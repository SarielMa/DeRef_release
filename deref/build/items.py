"""Per-argument check inputs and drop-variant inputs of one candidate call (FC-RewardBench and FinTrace).

For every argument `k` of a call two inputs are made:
  argument check: "{state}\\nFunction: {name} — {function description}\\nParameter: {k} — {info}\\n"
                  "Proposed value for {k}: {value JSON}", question Q_VALUE;
  drop variant:   noul_context(state, the call's action text with argument k removed), question NOUL_Q.
`info` is the parameter's description (FC-RewardBench) or "type: …; required: …; example: …" (FinTrace), or
NOT_PARAM when the function definition lacks the parameter. These texts are the "old" item texts that the state
builder (`deref.build.states`) re-bases on the final state; their token counts take part in the 2,048-token rule.
"""
import json

Q_VALUE = "Is this the correct value for this parameter, given the question?"
NOUL_Q = "Is this tool call a correct next step for answering the question?"
PAIR_Q = "Which tool call is a correct next step for answering the question?"
NOT_PARAM = "(not a parameter of this function)"
NO_DESC = "(no description available)"
# re-wording of NOT_PARAM in argument-check inputs (FC-RewardBench, FinTrace)
NODESC_OLD, NODESC_NEW = " — (not a parameter of this function)\n", " — (no description available)\n"


def noul_context(state, action):
    """Context of the pointwise whole-call input: the state followed by the proposed call."""
    return f"{state}\nProposed call: {action}"


def split_call(action_c1):
    """(name, raw argument text, parsed arguments) of a single-line action `call name({...})`."""
    name = action_c1.split("(", 1)[0][len("call "):]
    raw = action_c1.split("(", 1)[1][:-1]
    return name, raw, json.loads(raw)


def call_items(call_id, state, name, fdesc, args, info_of, raw_args):
    """Argument-check and drop-variant inputs for one call -> (items, argument ids, drop ids, any arg not in def)."""
    items, pids, dids, not_in_def = [], [], [], False
    for k, v in args.items():
        info = info_of(k)
        not_in_def |= info == NOT_PARAM
        ctx = (f"{state}\nFunction: {name} — {fdesc}\nParameter: {k} — {info}\n"
               f"Proposed value for {k}: {json.dumps(v, ensure_ascii=False)}")
        items.append({"item_id": f"{call_id}|p|{k}", "context": ctx, "question": Q_VALUE})
        pids.append(f"{call_id}|p|{k}")
        rest = {kk: vv for kk, vv in args.items() if kk != k}
        act = f"call {name}({json.dumps(rest)}) — {fdesc}"
        items.append({"item_id": f"{call_id}|d|{k}", "context": noul_context(state, act), "question": NOUL_Q})
        dids.append(f"{call_id}|d|{k}")
    return items, pids, dids, not_in_def


def write_items(out_dir, items, calls):
    """items.jsonl (ensure_ascii=False) and calls.jsonl (ASCII escapes) of one dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "items.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in items)
    with open(out_dir / "calls.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in calls)


def reword_not_param(items_path, out_path):
    """Replace NOT_PARAM by NO_DESC in argument-check inputs; write only the changed items (same item_id)."""
    out = []
    with open(items_path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if NODESC_OLD in x["context"]:
                assert x["context"].count(NODESC_OLD) == 1 and "|p|" in x["item_id"]
                out.append({**x, "context": x["context"].replace(NODESC_OLD, NODESC_NEW)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in out)
    return len(out)


def read_jsonl(path):
    """Read a UTF-8 JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    """Write dicts as UTF-8 JSONL with ensure_ascii=False."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
