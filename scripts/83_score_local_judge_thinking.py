"""Qwen3.8-27B judge with thinking on (Table 4): stage 1 samples the thinking (vLLM), stage 2 reads P(Yes) after it
(transformers). See deref/score/thinking_judge.py for the exact prompt, decoding and reading.

Sub-commands (W = a work directory):
  dedup  --items <f1.jsonl> [<f2.jsonl> ...] --work W   distinct (context, question) inputs -> W/uniq.jsonl, W/map.json
  shard  --work W --n N                                 length-balanced shards -> W/shard<i>.jsonl
  gen    --work W --shard i                             stage 1 (GPU, vLLM) -> W/think_<i>.jsonl
  score  --work W --shard i                             stage 2 (GPU, transformers) -> W/score_<i>.jsonl
  expand --work W --items <f.jsonl> --out <scores_dir>/<dataset>/<input>/qwen3_8_27b_thinking.jsonl
         scores of every item of one items file (rows {"item_id", "score", ...})
  check  --work W                                       CPU: thinking-on and thinking-off prompts differ only in the tail
As run for the paper (Table 4): the CFB inputs plain, plain_full and deref_full (plus deref) were deduplicated together,
split into two shards, and each shard took about 50 minutes on one B200 (stage 1 + stage 2).

  python scripts/83_score_local_judge_thinking.py dedup --items <inputs>/plain_cfb.jsonl <inputs>/plain_full_cfb.jsonl \\
      <inputs>/deref_full_cfb.jsonl <inputs>/deref_cfb.jsonl --work <work>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deref.common import read_jsonl, write_jsonl  # noqa: E402
from deref.score import thinking_judge as TJ  # noqa: E402

SCORE_NAME = "qwen3_8_27b_thinking"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("dedup", "shard", "gen", "score", "expand", "check"))
    ap.add_argument("--work", required=True, help="work directory")
    ap.add_argument("--items", nargs="+", help="items files (dedup: all inputs; expand: one file)")
    ap.add_argument("--n", type=int, default=2, help="number of shards")
    ap.add_argument("--shard", default="0")
    ap.add_argument("--out", help="expand: output score file")
    ap.add_argument("--batch", type=int, default=8, help="stage 2: at most this many rows per batch")
    ap.add_argument("--budget", type=int, default=40000, help="stage 2: padded-token budget per batch")
    ap.add_argument("--model-path", default=None, help="local model directory instead of the pinned HF id + revision")
    a = ap.parse_args()
    W = Path(a.work)
    if a.cmd == "dedup":
        uniq, mp = TJ.dedup(a.items)
        write_jsonl(W / "uniq.jsonl", uniq)
        json.dump(mp, open(W / "map.json", "w"), ensure_ascii=False)
        print(f"[dedup] {sum(len(m) for m in mp.values())} items -> {len(uniq)} distinct inputs")
    elif a.cmd == "shard":
        U = read_jsonl(W / "uniq.jsonl")
        order = sorted(range(len(U)), key=lambda i: -len(U[i]["context"]))
        for s in range(a.n):
            write_jsonl(W / f"shard{s}.jsonl", [U[i] for i in order[s::a.n]])
            print(f"shard{s}: {len(order[s::a.n])} inputs")
    elif a.cmd == "gen":
        TJ.generate(read_jsonl(W / f"shard{a.shard}.jsonl"), W / f"think_{a.shard}.jsonl", a.model_path)
    elif a.cmd == "score":
        items = read_jsonl(W / f"shard{a.shard}.jsonl")
        think = {r["item_id"]: r for r in read_jsonl(W / f"think_{a.shard}.jsonl")}
        write_jsonl(W / f"score_{a.shard}.jsonl", TJ.score(items, think, a.batch, a.budget, a.model_path))
    elif a.cmd == "expand":
        assert a.items and len(a.items) == 1 and a.out, "expand needs one --items file and --out"
        mp = json.load(open(W / "map.json"))[Path(a.items[0]).name]
        sc = {}
        for f in sorted(W.glob("score_*.jsonl")):
            sc.update({r["item_id"]: r for r in read_jsonl(f)})
        missing = [k for k, u in mp.items() if u not in sc]
        assert not missing, f"{len(missing)} items without a score, e.g. {missing[:3]}"
        write_jsonl(a.out, [dict(sc[u], item_id=k) for k, u in mp.items()])
        print(f"[expand] {len(mp)} scores -> {a.out}")
    else:
        tok = TJ.load_tokenizer(a.model_path)
        U = read_jsonl(W / "uniq.jsonl")
        bad = sum(not TJ.check_render(tok, x) for x in U)
        print(f"[check] {len(U)} inputs; thinking-on / off renders differing outside the tail: {bad}")
        assert bad == 0


if __name__ == "__main__":
    main()
