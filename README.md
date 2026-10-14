# DeRef

Code for the paper **"Judges Follow the Reference: The Derivation Gap in Verifying Tool-Call Arguments"**.

The code rebuilds every table and figure in the main body of the paper. The pipeline has five stages:

1. **build**: build the evaluation data from three public benchmarks (FC-RewardBench, ComplexFuncBench, FinTrace).
2. **generate**: write prompts and generate the references with vLLM.
3. **inputs**: build the judge inputs (Plain, full history, DeRef, and the baselines).
4. **score**: score the inputs with CLM, local judges (Qwen3, Llama, Mistral) and Jev.
5. **stats**: compute the numbers in the tables and figures.

The data from stage 1 is released on Hugging Face as [`lm2445/DeRef`](https://huggingface.co/datasets/lm2445/DeRef),
so you can start at stage 2.
No model outputs or results are included. Every number is recomputed from the pipeline.

## Setup

```
pip install -r requirements.txt          # main environment (Python 3.11, GPU)
pip install -r requirements-build.txt    # only for building the data from scratch (Python 3.12)
```

- Generation and local-judge scoring need a GPU. Qwen3.8-27B needs a large-memory GPU. Everything else runs on CPU.
- Jev is a hosted judge. Set `JEV_KEY` to use it.
- CLM needs the `clm` package and the checkpoint `CLM_v0.1-8B.pt`.

Run all commands from this folder. Every script shows its options with `--help`.

## Folder names used below

| name | meaning |
|---|---|
| `D` | data folder (the one that contains `data/`) |
| `G` | prompts and generations |
| `I` | judge inputs |
| `S` | scores |
| `T` | output tables |

## Which script makes what

| paper item | build inputs | compute numbers |
|---|---|---|
| Table 1 | 10, then score | 50, 54 |
| Table 2 | 10, 12, 13, 20, 22, 80–82, then score | 51, 52, 86, 54 |
| Table 3 | 13, 20, 22, then score | 53, 54 |
| Table 4 | 12, 14, 83 | 84 |
| Figure 1 | 10, 13, 20, 22 | 85 |
| Figure 2a | 60–63, 22 | 64 |
| Figure 2b | 13, 70 | 71 |
| Figure 3 | 13, 72 | 73 |

## How to run

### 1. Generate references

```
python scripts/20_build_prompts.py --data-dir D --out-dir G/prompts
python scripts/22_generate.py --prompts G/prompts/deref.jsonl --out G/generations/deref.jsonl --mode thinking --max-new 8192
```

Run `22_generate.py` once for each prompt file you need (`deref`, `revise`, `evidence_search`, ...).

### 2. Build judge inputs

```
python scripts/10_build_plain.py --data-dir D --out I
python scripts/12_build_plain_full.py --data-dir D --out I
python scripts/13_build_reference_items.py --method deref --data-dir D --gen-dir G --out I
python scripts/14_build_deref_full.py --data-dir D --gen-dir G --out I
```

`--method` can be `deref`, `revise`, `evidence_search`, `sgv_prior`, `deref_nothink`, `deref_qwen3_8b` or `expected_value`.

### 3. Score

| judge | command |
|---|---|
| CLM | `CLM_CKPT=<checkpoint> bash scripts/41_score_clm.sh <list.tsv>` |
| Qwen3, Llama, Mistral | `JUDGES="qwen3_8b qwen3_8_27b" bash scripts/42_score_local_judges.sh <list.tsv>` |
| local judges, full history | `bash scripts/43_score_local_judge_long.sh <list.tsv>` |
| Jev | `44_jev_dedup.py`, then `45_score_jev.py`, then `47_jev_final.py` |

Each line of `list.tsv` is `<input file><TAB><output file>`.
Scores go to `S/<dataset>/<input>/<judge>.jsonl`.

### 4. Tables

```
python scripts/50_table1_gap.py    --data-dir D --scores-dir S --out-dir T
python scripts/51_argument_grid.py --data-dir D --scores-dir S --out-dir T
python scripts/86_table2_self_reference.py --data-dir D --scores-dir S --out-dir T
python scripts/52_judge_averages.py --data-dir D --scores-dir S --out-dir T
python scripts/53_ablation.py      --data-dir D --scores-dir S --out-dir T
python scripts/84_table4.py        --data-dir D --scores-dir S --out-dir T
python scripts/54_paper_tables.py  --in-dir T --out-dir T
```

### 5. Figures

```
python scripts/85_figure1_case.py --data-dir D --scores-dir S --deref-items I/deref_cfb.jsonl
python scripts/64_figure2a.py --data-dir D --scores-dir S --out-dir T
python scripts/71_figure2b.py --data-dir D --inputs-dir I --scores-dir S --out-dir T
python scripts/73_figure3.py  --data-dir D --inputs-dir I --scores-dir S --out-dir T
```

Each figure first needs its own inputs and scores (see the table "Which script makes what").

## Building the data from scratch (optional)

Scripts `01`–`09` rebuild `D/data` from the original benchmark files. You only need this if you do not use the
released data. Run them in order; `--help` lists the source files each one needs.

## Code layout

```
deref/build      build the data            scripts 01-09, 60-62
deref/generate   prompts and generation    scripts 20, 22, 80, 81
deref/inputs     judge inputs              scripts 10-14, 63, 70, 72, 82
deref/score      judges                    scripts 40-47, 83
deref/stats      tables and figures        scripts 50-54, 64, 71, 73, 84-86
```

## Citation

```bibtex
@misc{ma2026judgesfollowreferencederivation,
      title={Judges Follow the Reference: The Derivation Gap in Verifying Tool-Call Arguments}, 
      author={Linhai Ma and Yan Wang and Lingfei Qian and Yupeng Cao and Victor Gutierrez Basulto},
      year={2026},
      eprint={2610.18497},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.18497}, 
}
```
