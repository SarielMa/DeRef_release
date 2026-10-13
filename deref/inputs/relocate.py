"""Judge inputs and DeRef prompts of FCRB-Relocate (Figure 2a).

Records (`data/fcrb_relocate.jsonl`) have the FC-RewardBench record layout, one per state, version and candidate
role; the state of a version has one earlier call (`find_saved_items`) whose return is shown cut to 1,000 characters.
Only the relocated argument is judged (`in_analysis`). All texts are rendered with the FC-RewardBench rules:
  plain       Plain input (state + argument tail; no state is over 2,048 CLM tokens, so no ladder rung is used);
  plain_full  Full history: the uncut return (32,768-token ladder of `full_text`, never needed here);
  deref       DeRef prompt per state and version (the derivation prompt of `generate.prompts`, Qwen3.8-27B, thinking on;
              one reference shared by the three candidate values), and the judge input: state + HEAD + reference + tail.
              The state is not laddered: an input over 2,048 CLM tokens gets no item and keeps its Plain score."""
from deref.common import read_jsonl
from deref.generate import prompts as GP
from deref.inputs import full_text as FT
from deref.inputs import plain as PL
from deref.inputs import reference as RF
from deref.inputs import state as ST

DATASET = "fcrb_relocate"


def load_records(data_dir):
    """Records of `<data_dir>/data/fcrb_relocate.jsonl` (or `<data_dir>/fcrb_relocate.jsonl`), as FC-RewardBench views
    (`dataset` = "fcrb", which selects the FC-RewardBench schema text)."""
    from pathlib import Path
    d = Path(data_dir)
    p = d / "data" / f"{DATASET}.jsonl"
    return [{**r, "dataset": "fcrb"} for r in read_jsonl(p if p.exists() else d / f"{DATASET}.jsonl")]


def plain_rows(records):
    return PL.build(records)


def plain_full_rows(records, builder):
    return [FT.plain_full_item(builder, r, a) for r, a in ST.arguments(records)]


def deref_prompts(records, tok27):
    """(prompt rows, map rows) of the DeRef derivation (thinking on)."""
    return GP.build(records, DATASET, tok27, methods=("deref",), log=lambda *_: None)["deref"]


def deref_rows(records, generations, item_map, tok8):
    """(rows, fallbacks): DeRef judge inputs (item id `<argument item_id>|deref`); fallbacks keep their Plain score."""
    rows, fallbacks = [], []
    for rec, arg in ST.arguments(records):
        out = generations.get(item_map.get(arg["item_id"]))
        a = None if out is None else RF.answer(out)
        if a is None:
            fallbacks.append((arg["item_id"], "no reference"))
            continue
        S, lab, tail, q = PL.base_parts(rec, arg)
        ctx = S + RF.HEAD + a.strip() + tail
        n = ST.clm_tokens(tok8, ctx, q)
        if n > ST.LIMIT:
            fallbacks.append((arg["item_id"], "over 2048"))
            continue
        rows.append({"item_id": f"{arg['item_id']}|deref", "base_item": arg["item_id"], "context": ctx, "question": q,
                     "clm_tokens": n})
    return rows, fallbacks
