"""Figure 1: the ComplexFuncBench history-only example (Plain vs DeRef scores of the four Table 2 judges).

Selection rule: correct ComplexFuncBench arguments on the history_only path whose DeRef reference states a definite
value (expected-value parser) equal to the gold value, whose gold value does not occur in the Plain input (the
`Proposed value for` line excluded), and that all four judges reject under Plain (score < 0.5 and < tau) and accept
under DeRef (score >= 0.5 and >= tau); candidates are ordered by Plain input length, shortest first. tau is each judge's
acceptance threshold for that input (80% of the non-placeholder correct arguments accepted). The figure shows a fixed
example (`--example`, default Car-Rental-104|k3|p|0:vehicle_id) as long as it satisfies the rule; otherwise the first
candidate is shown.
Reads the data package, the score files (`<scores>/cfb/{plain,deref}/<judge>.jsonl`) and the DeRef judge items of
ComplexFuncBench (13_build_reference_items.py --method deref, file deref_cfb.jsonl). Prints the selection counts and
the example (scores rounded in the figure to two decimals) and writes them to --out (JSON) if given.

  python scripts/85_figure1_case.py --data-dir <deref_data> --scores-dir <scores> --deref-items <inputs>/deref_cfb.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import read_jsonl  # noqa: E402
from deref.inputs import expected_value_parser as EVP  # noqa: E402
from deref.inputs import plain as PL  # noqa: E402
from deref.inputs import reference as RF  # noqa: E402
from deref.inputs import state as ST  # noqa: E402
from deref.stats import core, load  # noqa: E402

D = "cfb"
EXAMPLE = "Car-Rental-104|k3|p|0:vehicle_id"
TRAIL = ".,;:"


def norm(x):
    return " ".join(str(x).strip().strip("\"'`").lower().split())


def norm_value(x):
    """Strip surrounding whitespace, trailing . , ; : and quotes / backticks repeatedly; lower case; single spaces."""
    s = str(x)
    while True:
        t = s.strip().rstrip(TRAIL).strip().strip("\"'`")
        if t == s:
            break
        s = t
    if not s:
        return norm(x)
    return " ".join(s.lower().split())


def same_value(stated, value):
    """Stated reference value equals the argument value (normalised text, or equal as numbers)."""
    a = norm_value(stated)
    b = norm_value(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def reference_text(context):
    """The reference block of a DeRef judge input (between its header and the Function line)."""
    i = context.index(RF.HEAD) + len(RF.HEAD)
    return context[i:context.index("\nFunction: ", i)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="data package directory")
    ap.add_argument("--scores-dir", required=True, help="score files <scores_dir>/<dataset>/<input>/<judge>.jsonl")
    ap.add_argument("--deref-items", required=True, help="DeRef judge items of ComplexFuncBench (deref_cfb.jsonl)")
    ap.add_argument("--example", default=EXAMPLE, help="example kept if it satisfies the rule")
    ap.add_argument("--out", default=None, help="write the selection and the example as JSON")
    a = ap.parse_args()
    st = load.load_sets(a.data_dir)[D]
    SC, _ = load.load_scores(a.scores_dir, ("plain", "deref"), load.J4)
    sc = {(j, i): SC[(D, j)][i] for j in load.J4 for i in ("plain", "deref")}
    cal = [k for k in st.correct if st.path[k] != "placeholder"]
    ci_cal = np.array([st.cidx[k] for k in cal])
    tau = {key: float(core.tau_q_draws(load.arr(s, cal), ci_cal, st.counts)[0]) for key, s in sc.items()}
    refs = {r["base_item"]: reference_text(r["context"]) for r in read_jsonl(a.deref_items)}
    recs = ST.load_records(a.data_dir, D)
    arg = {x["item_id"]: (r, x) for r in recs for x in r["arguments"]}
    plain = {k: PL.plain_item(*arg[k])["context"] for k in st.correct}

    def ref_gold(k):
        r = EVP.prior_value12(refs.get(k, ""))
        return r["definite"] and same_value(r["value"], arg[k][1]["value"])

    def passes(k):
        return all(sc[(j, "plain")][k] < 0.5 and sc[(j, "plain")][k] < tau[(j, "plain")] and
                   sc[(j, "deref")][k] >= 0.5 and sc[(j, "deref")][k] >= tau[(j, "deref")] for j in load.J4)

    def not_in_plain(k):
        v = arg[k][1]["value"]
        vt = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return vt not in "\n".join(l for l in plain[k].split("\n") if not l.startswith("Proposed value for "))

    c0 = sorted(k for k in st.correct if st.path[k] == "history_only")
    c1 = [k for k in c0 if ref_gold(k)]
    c2 = [k for k in c1 if not_in_plain(k)]
    c3 = sorted((k for k in c2 if passes(k)), key=lambda k: (len(plain[k]), k))
    keep = a.example in c3
    ex = a.example if keep else c3[0]
    rec, x = arg[ex]
    v = x["value"] if isinstance(x["value"], str) else json.dumps(x["value"], ensure_ascii=False)
    src = next(((p["label"], len(p["text"]), p["text"].index(v)) for p in rec["history"]
                if p["kind"] == "return" and v in p["text"]), None)
    out = {"history_only correct": len(c0), "reference value = gold": len(c1), "gold not in Plain input": len(c2),
           "all four judges reject under Plain and accept under DeRef": len(c3), "candidates by Plain length": c3,
           "example": ex, "example kept": keep, "value": x["value"],
           "first return holding the value (label, characters, position)": src,
           "reference": refs.get(ex),
           "scores": {load.JUDGE_NAMES[j]: {"plain": sc[(j, "plain")][ex], "plain_tau": tau[(j, "plain")],
                                            "deref": sc[(j, "deref")][ex], "deref_tau": tau[(j, "deref")]}
                      for j in load.J4}}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    for j, s in out["scores"].items():
        print(f"{j:12s} Plain {s['plain']:.2f}  DeRef {s['deref']:.2f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
