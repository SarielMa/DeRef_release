"""Loaders for the statistics step: argument-level analysis sets (from the data package) and per-item judge scores.

Score files: `<scores_dir>/<dataset>/<input>/<judge>.jsonl`, rows `{"item_id": ..., "score": float}`.
- `plain` is the base. Every other input is a dict over the same argument ids; an argument without its own score keeps
  the Plain score.
- Reference-reading inputs key their items as `<argument item_id>|<tag>` (tag per input in `TAG`); the full-text inputs
  (`plain_full`, `deref_full`) use the bare argument id.
- FC-RewardBench has no truncation: `plain_full` and `deref_full` are the SAME objects as `plain` and `deref` (pairs of
  identical objects are not compared). CLM has no full-text inputs. FC-RewardBench has no history, so no value search.
- An input whose file does not exist is reported in the `status` column instead of being scored.
"""
import json
from pathlib import Path

import numpy as np

from deref.stats import core

DATASETS = ("fcrb", "cfb", "fintrace")
DISPLAY = {"fcrb": "FC-RewardBench", "cfb": "ComplexFuncBench", "fintrace": "FinTrace"}
JUDGE_NAMES = {"clm": "CLM", "jev": "Jev", "qwen3_4b": "Qwen3-4B", "qwen3_8b": "Qwen3-8B", "qwen3_8_27b": "Qwen3.8-27B", "llama3_1_8b": "Llama-3.1-8B-Instruct",
               "mistral_7b": "Mistral-7B-Instruct-v0.3"}
# judges of Table 2 (argument grid, paired differences, judge averages)
J4 = ("clm", "jev", "qwen3_8b", "qwen3_8_27b")
# Table 2 averages are also reported without Jev
J3 = ("clm", "qwen3_8b", "qwen3_8_27b")
# judges of Table 1
J7 = ("clm", "jev", "qwen3_4b", "qwen3_8b", "qwen3_8_27b", "llama3_1_8b", "mistral_7b")
# judges whose context is too short for the full-text inputs
NO_FULL_TEXT = ("clm",)
# input -> item-id tag of its reference-reading items ("" = bare argument id)
TAG = {"plain_full": "", "evidence_search": "evidence_search", "sgv_prior": "sgv_prior", "revise": "revise",
       "deref": "deref", "deref_full": "", "deref_nothink": "deref_nothink", "deref_qwen3_8b": "deref_qwen3_8b",
       "expected_value": "expected_value"}
PATHS = ("visible", "history_only", "format", "default", "no_source")
NONVIS = core.NONVIS

ST_OK = "ok"
ST_MISSING = "missing"
ST_NA = "not applicable"
ST_SAME_PLAIN = "same as plain (FC-RewardBench has no truncation)"
ST_SAME_DEREF = "same as deref (FC-RewardBench has no truncation)"


# ---------------------------------------------------------------- analysis sets
class ArgSet:
    """Argument-level analysis set of one dataset.

    correct / wrong: sorted argument ids; path: {id: path}; cidx: {id: index of its query_id in the sorted cluster list};
    counts: B x n_clusters bootstrap cluster counts (Gap_tau)."""

    def __init__(self, name, correct, wrong, path, cidx, n_clusters):
        self.name = name
        self.correct, self.wrong = correct, wrong
        self.path, self.cidx = path, cidx
        self.counts = core.cluster_counts(n_clusters)


def load_sets(data_dir):
    """{dataset: ArgSet} from `<data_dir>/<dataset>.jsonl` and `<data_dir>/bootstrap_clusters.json`.

    The cluster of an argument is the query_id of its record (wrong calls share the query_id of their state). Clusters are
    referred to by their position in the shipped sorted list, which keeps the resampling order of the original ids."""
    data_dir = Path(data_dir)
    if (data_dir / "data" / "bootstrap_clusters.json").exists():  # the package root was given
        data_dir = data_dir / "data"
    with open(data_dir / "bootstrap_clusters.json", encoding="utf-8") as f:
        universe = json.load(f)
    out = {}
    for d in DATASETS:
        clusters = universe[d]
        assert clusters == sorted(clusters) and len(set(clusters)) == len(clusters), f"{d}: cluster list not sorted / unique"
        pos = {c: i for i, c in enumerate(clusters)}
        correct, wrong, path, cidx = [], [], {}, {}
        with open(data_dir / f"{d}.jsonl", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                for a in rec["arguments"]:
                    if not a["in_analysis"]:
                        continue
                    k = a["item_id"]
                    (correct if a["label"] == "correct" else wrong).append(k)
                    path[k] = a["path"]
                    cidx[k] = pos[rec["query_id"]]
        assert len(set(correct)) == len(correct) and len(set(wrong)) == len(wrong) and not set(correct) & set(wrong), d
        out[d] = ArgSet(d, sorted(correct), sorted(wrong), path, cidx, len(clusters))
    return out


# ---------------------------------------------------------------- scores
def read_scores(path):
    """{item_id: score} of one score file, or None if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["item_id"]] = r["score"]
    return out


def over_plain(src, tag, plain):
    """{argument id: own score, else Plain score} over the Plain items."""
    suffix = f"|{tag}" if tag else ""
    return {k: src.get(f"{k}{suffix}", v) for k, v in plain.items()}


def load_scores(scores_dir, inputs, judges=J4):
    """SC[(dataset, judge)] = {input: {argument id: score}}, NOTE[(dataset, judge, input)] = status text.

    Identity matters: on FC-RewardBench `plain_full` IS `plain` and `deref_full` IS `deref` (same dict objects)."""
    scores_dir = Path(scores_dir)
    SC, NOTE = {}, {}
    for d in DATASETS:
        for j in judges:
            plain = read_scores(scores_dir / d / "plain" / f"{j}.jsonl")
            if plain is None:
                SC[(d, j)] = {}
                for inp in inputs:
                    NOTE[(d, j, inp)] = ST_MISSING
                continue
            s = {"plain": plain}
            for inp in inputs:
                if inp == "plain" or inp in ("plain_full", "deref_full"):
                    continue
                src = read_scores(scores_dir / d / inp / f"{j}.jsonl")
                if src is None:
                    NOTE[(d, j, inp)] = ST_MISSING
                    continue
                s[inp] = over_plain(src, TAG[inp], plain)
            for inp, base, note in (("plain_full", "plain", ST_SAME_PLAIN), ("deref_full", "deref", ST_SAME_DEREF)):
                if inp not in inputs:
                    continue
                if j in NO_FULL_TEXT:
                    NOTE[(d, j, inp)] = ST_NA
                elif d == "fcrb":
                    if base in s:
                        s[inp] = s[base]
                        NOTE[(d, j, inp)] = note
                    else:
                        NOTE[(d, j, inp)] = ST_MISSING
                else:
                    src = read_scores(scores_dir / d / inp / f"{j}.jsonl")
                    if src is None:
                        NOTE[(d, j, inp)] = ST_MISSING
                    else:
                        s[inp] = over_plain(src, TAG[inp], plain)
            SC[(d, j)] = s
    return SC, NOTE


def plain_scores(scores_dir, d, j):
    """Plain scores of one judge (Table 1), or None."""
    return read_scores(Path(scores_dir) / d / "plain" / f"{j}.jsonl")


def arr(s, keys):
    """Scores of `keys` as a float array."""
    return np.array([s[k] for k in keys], float)
