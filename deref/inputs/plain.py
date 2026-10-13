"""Plain judge inputs: one per argument, `context` = Plain state (stored rung) + argument tail, `question` = Q_VALUE.
These are the texts every judge scores as "Plain" (the model-specific prompt wrapping is applied by the scorers)."""
from deref.inputs import state as ST


def base_parts(rec, arg):
    """(Plain state, rung label, argument tail, question) of one argument; the Plain context is state + tail."""
    lab, S = ST.plain_state(rec)
    return S, lab, ST.argument_tail(rec, arg), ST.Q_VALUE


def plain_item(rec, arg):
    S, _, tail, q = base_parts(rec, arg)
    return {"item_id": arg["item_id"], "context": S + tail, "question": q}


def build(records, analysis_only=True):
    """Plain rows of the records' arguments (default: only the argument-level analysis set)."""
    return [plain_item(r, a) for r, a in ST.arguments(records, analysis_only)]
