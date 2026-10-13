# DeRef: code for "Judges Follow the Reference: The Derivation Gap in Verifying Tool-Call Arguments"

Anonymous authors. This package contains the code behind every table and figure in the main body of the paper, from
the three public benchmarks to the cells of the tables:

1. **build** the data package (the argument-level analysis sets of FC-RewardBench, ComplexFuncBench and FinTrace, and
   the controlled set FCRB-Relocate);
2. **inputs**: build the judge inputs (Plain, full history, Schema facts, and the inputs that read a reference);
3. **generate**: build the derivation / baseline prompts and generate the references with vLLM;
4. **score**: score the judge inputs with CLM, local Yes/No judges (Qwen3, Llama, Mistral) and Jev;
5. **stats**: tau, Gap_tau, argument-level AUROC, query-clustered bootstrap, paired differences, and the tables.

The data package (`deref_data/`, released on the Hugging Face Hub as
[`lm2445/DeRef`](https://huggingface.co/datasets/lm2445/DeRef)) is read from a local directory.
No model outputs, scores or results are included; every number is recomputed from the scores the pipeline produces.
All paths are command-line arguments; every entry script prints its usage with `--help`.

## Paper item -> scripts

| paper item | content | inputs / generation | statistics |
|---|---|---|---|
| Table 1 | Gap_tau with Plain input, seven judges | 10 (Plain); scoring 41, 42, 44–47 | 50, 54 |
| Figure 1 | one ComplexFuncBench history-only case, Plain vs DeRef | 10, 20 + 22 (deref), 13 | 85 |
| Figure 2a | FCRB-Relocate, argument AUROC | 60–62 (set), 63, 22 | 64 |
| Figure 2b | FC-RewardBench correct defaults, acceptance at tau | 70 (Schema facts), 13 (deref) | 71 |
| Table 2 | argument AUROC of four judges per input; DeRef - RevisEval | 10, 12, 13 (deref, revise, evidence_search), 20 + 22, 80–82 (self-generated reference) | 51, 86, 54; judge averages 52 |
| Table 3 | ablation on ComplexFuncBench, Qwen3.8-27B | 13 (sgv_prior, deref_qwen3_8b, deref_nothink, expected_value), 20 + 22 | 53, 54 |
| Table 4 | reference x judge thinking, ComplexFuncBench, Qwen3.8-27B, full history | 12, 14; thinking-on scoring 83 | 84 |
| Figure 3 | reference swap | 13 (deref), 72 | 73 |

Input names used by the code: `plain` = Plain, `plain_full` = Full history, `evidence_search` = ProvenanceGuard
evidence, `revise` = RevisEval, `deref` = DeRef, `self_reference` = Self-generated reference, `deref_full` = DeRef with
the full history (Table 4), `schema_facts` = Schema facts (Figure 2b), `deref_swap` = DeRef with the swapped value
(Figure 3); Table 3 variants: `sgv_prior` = w/o full history (the deriver reads the truncated judge input),
`deref_qwen3_8b` = Qwen3-8B as deriver, `deref_nothink` = deriver thinking off, `expected_value` = parsed expected value
only.

## Environment


| environment | used by | requirements file |
|---|---|---|
| Python 3.11.16, CUDA 13.0 | steps 03, 08 (tokenizer), 10–14, 20–22, 40–47, 50–54, 60–86 | `requirements.txt` |
| Python 3.12.11 | steps 01, 02, 05, 07, 09 (`datasets`, `sentence-transformers`) | `requirements-build.txt` |

Steps 50–54, 64, 71, 73, 84–86 also run (with identical numbers) in the build environment. Hardware: generation and local-judge scoring
ran on one NVIDIA B200 (183 GB) per job; Qwen3.8-27B needs a large-memory GPU. Everything else is CPU.
Jev is a hosted judge (OpenRouter, pinned model `typesafe/jev-1.13-20260917`); set `JEV_KEY` (and optionally
`JEV_BASE_URL`, default `https://openrouter.ai/api`).

Models (Hugging Face id @ revision, pinned in `deref/score/models.py` and `deref/generate/models.py`):
Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218 (judge, CLM encoder, tokenizer for all 2,048-token rules,
Table 3 deriver), Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 (deriver and judge),
Qwen/Qwen3-4B @ 1cfa9a7208912126459214e8b04321603b3df60c, meta-llama/Llama-3.1-8B-Instruct @
0e9e39f249a16976918f6564b8830bc894c89659, mistralai/Mistral-7B-Instruct-v0.3 @ c170c708c41dac9275d15a8fff4eca08d52bab71,
BAAI/bge-large-en-v1.5 (FinTrace tool similarity, step 05). CLM: the `clm` package (`contrastive-lm` 0.1.0) with the
head checkpoint `CLM_v0.1-8B.pt` (sha256 b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5).

Run every command from this directory (`python scripts/<step>.py ...`).

## Sources (step 1 only)

| source | version |
|---|---|
| `ibm-research/fc-reward-bench` (HF dataset, `save_to_disk` copy) | revision 269929c3329e603e87ed3203de42896cc03ddbf3 |
| `gorilla-llm/Berkeley-Function-Calling-Leaderboard` BFCL v3 questions / possible answers (HF dataset) | revision 61fc0608cfd831fcfbbaa676ebdfef0ed963eeda |
| BFCL AST checker (gorilla GitHub repository, `berkeley-function-call-leaderboard/bfcl`) | commit 70b6a4a21445 |
| `zai-org/ComplexFuncBench` `ComplexFuncBench.jsonl` (HF dataset) | no revision recorded; file sha256 be1e0f5951b666f6543b946e915a431a39bbda1807481878eb331245276ac088 |
| `AgentSuite/complex-func-bench-trajectories` (HF dataset, one file per model) | revision fcc906da345ec46d8815792bfff4724d7cca86b0 |
| AgentSuite ComplexFuncBench audit lists `cfbench_filter_ids.csv`, `cfbench_issues.csv` | origin / revision not recorded; sha256 in the data card |
| `YupengCao/FinTrace` `evaluation/testset.json` (HF dataset) | no revision recorded; file sha256 b10f3edcfe5df13b9d3aa9cc4dccb90dcc523234ee0c19cee5cd12c68055eb0a |
| FMP stable API documentation pages (FinTrace tool descriptions) | scraped 2026-09-30 (`scripts/04_fetch_fmp_docs.py`); a new scrape is not guaranteed to be identical |

`deref/build/config/fcrb_system_prompt_only_tasks.json` is a hand-made list of 35 FC-RewardBench tasks whose user turn
is a system prompt; they are removed in step 07.

## Pipeline

`W` = build work directory, `D` = data package directory (the folder that contains `data/`; scripts also accept `D/data`),
`I` = judge-input directory, `G` = generation directory (`G/prompts`, `G/generations`), `S` = score directory,
`T` = table directory, `R` = FCRB-Relocate work directory.
Score files are `S/<dataset>/<input>/<judge>.jsonl` with rows `{"item_id", "score"}`; reference-reading items carry
`<argument item_id>|<method>` and arguments without their own item keep their Plain score.

### Step 1: data package (CPU; about 1.5 hours in total, dominated by step 08 on FinTrace)

| step | command | reads → writes |
|---|---|---|
| 01 | `python scripts/01_build_fcrb.py --raw-dir <fc-reward-bench save_to_disk dir> --work-dir W` | source rows → `W/fcrb/{processed,items,items_nodesc}` |
| 02 | `python scripts/02_label_fcrb_bfcl.py --bfcl-dir <BFCL v3 dir> --gorilla-dir <gorilla checkout> --work-dir W` | BFCL v3 + checker → `W/fcrb/bfcl_param_labels.jsonl` |
| 03 | `python scripts/03_build_cfb.py --gold <ComplexFuncBench.jsonl> --trajectories <dir> --agentsuite-dir <dir> --work-dir W` | → `W/cfb/...` (Qwen3-8B tokenizer; 5–10 min) |
| 04 | `python scripts/04_fetch_fmp_docs.py --out-dir <fmp_docs dir>` | network scrape (or use the cached pages) |
| 05 | `python scripts/05_build_fintrace_tools.py --testset <testset.json> --fmp-docs <fmp_docs dir> --work-dir W` | → `W/fintrace/{tools.jsonl,tool_sim.npz}` |
| 06 | `python scripts/06_build_fintrace.py --testset <testset.json> --work-dir W` | → `W/fintrace/...` (wrong calls with seed 20260930) |
| 07 | `python scripts/07_export_states.py --fcrb-raw <dir> --cfb-gold <file> --testset <file> --agentsuite-dir <dir> --work-dir W` | cleaning, history, window, labels → `W/states/export.jsonl` |
| 08 | `python scripts/08_build_states.py --work-dir W` | 2,048-token ladder over every judge input of a state → `W/states/<dataset>/...` |
| 09 | `python scripts/09_build_package.py --fcrb-raw <dir> --cfb-gold <file> --work-dir W --out-dir D/data` | paths, CFB parallel-call drop, analysis sets → `D/data/{fcrb,cfb,fintrace}.jsonl`, `bootstrap_clusters.json` |

(Exact arguments: `--help` of each script.)

### Step 2: generation prompts and references (prompts on CPU, generation on GPU)

```
python scripts/20_build_prompts.py --data-dir D --out-dir G/prompts            # all methods; ~25 min CPU
python scripts/22_generate.py --prompts G/prompts/deref.jsonl --out G/generations/deref.jsonl --mode thinking --max-new 8192
```
As run for the paper (Qwen3.8-27B unless noted; vLLM 0.30.0, bf16, one GPU):

| prompt file | mode | max new tokens | GPU time of the paper run |
|---|---|---|---|
| deref, sgv_prior, revise, evidence_search | thinking (T=1.0, top-p 0.95, top-k 20, seed 20260930, context 40,960) | 8,192 | 15–50 min each |
| deref_qwen3_8b | thinking, `--model qwen3_8b` (T=0.6) | 8,192 | ~11 min |
| deref_nothink (Table 3 "deriver thinking off") | greedy (context 32,768 + new) | 1,024 | ~10 min |

The other `*_nothink` files are the thinking-off originals the thinking-on prompts are re-rendered from; they need not
be generated. Sampled generations are not guaranteed to be byte-identical under vLLM batching.

### Step 3: judge inputs (CPU)

```
python scripts/10_build_plain.py --data-dir D --out I                        # Plain, all judges (seconds)
python scripts/12_build_plain_full.py --data-dir D --out I                   # Full history (cfb, fintrace)
python scripts/13_build_reference_items.py --method deref --data-dir D --gen-dir G --out I
#   methods: deref revise evidence_search sgv_prior deref_nothink deref_qwen3_8b expected_value
python scripts/14_build_deref_full.py --data-dir D --gen-dir G --out I       # DeRef + full history (cfb, fintrace)
```
Runs of 12–14 tokenise tens of thousands of texts (minutes to about an hour on CPU for FinTrace).

### Step 4: scoring

| judge | command | inputs | GPU time of the paper runs |
|---|---|---|---|
| CLM | `CLM_CKPT=<CLM_v0.1-8B.pt> bash scripts/41_score_clm.sh <list.tsv>` (vLLM pooling server + `41_score_clm.py`) | all inputs of at most 2,048 tokens | ~45 min for the Plain inputs of the three datasets |
| Qwen3-8B / Qwen3.8-27B | `JUDGES="qwen3_8b qwen3_8_27b" bash scripts/42_score_local_judges.sh <list.tsv>` | short inputs | Plain: 3 h (8B), 5 h (27B) |
| Qwen3-4B, Llama-3.1-8B, Mistral-7B | same script, other `JUDGES` | Plain (Table 1) | 25–70 min each |
| Qwen3-8B / Qwen3.8-27B, full history | `bash scripts/43_score_local_judge_long.sh <list.tsv>` | plain_full, deref_full | 0.5–4 h per input |
| Jev | `python scripts/44_jev_dedup.py` -> `45_score_jev.py` -> `47_jev_final.py`; full history: `bash scripts/45_jev_full_text.sh` | Plain, Full history, the Table 2 reference inputs, Figure 1 / 2 / 3 inputs | API, a few hours |

`list.tsv` lines are `<items.jsonl><TAB><output>` (see each script). Run `scripts/40_check_lengths.py` before scoring
(the shell examples do). CLM: the Table 1 Plain run used the vLLM server's `--max-model-len 2048`; all later CLM runs
used 2064, which avoids a vLLM hang on inputs of exactly 2,048 tokens under concurrent requests (inputs are cut at
2,048 on the client in both cases). Jev: `45_score_jev.py --probe-max` is a local length check (Qwen3-8B tokens,
default 9,000); the full-history run passes `--probe-max 40000`, so Jev's own context limit decides and refused
inputs go down the deletion ladder (`46_jev_ladder_next.py`). Local-judge scores move slightly with batch composition
and CLM scores by up to ~0.03 between identical runs.

### Step 5: Tables 1-3 (CPU)

```
python scripts/50_table1_gap.py      --data-dir D --scores-dir S --out-dir T   # Table 1 (seven judges, Plain)
python scripts/51_argument_grid.py   --data-dir D --scores-dir S --out-dir T   # Table 2: per-judge AUROC / Gap_tau grid and paired differences (~15 min)
python scripts/86_table2_self_reference.py --data-dir D --scores-dir S --out-dir T   # Table 2: self-generated reference rows
python scripts/52_judge_averages.py  --data-dir D --scores-dir S --out-dir T   # averages over the four judges (Section 5.2)
python scripts/53_ablation.py        --data-dir D --scores-dir S --out-dir T   # Table 3 (Qwen3.8-27B)
python scripts/54_paper_tables.py    --in-dir T --out-dir T                    # paper views (x100, "est [lo, hi]")
```
Every CSV is written with proportions and as a `_x100` copy (two decimals). Statistics: tau accepts 80% of the
non-placeholder correct arguments (ties split by the proportion q); Gap_tau = acceptance of visible minus out-of-view
correct arguments; AUROC with ties counted as 1/2; 1,000 query-clustered bootstrap draws (seed 20260930) with 95%
percentile intervals; judge averages average the judges' paired draws draw by draw.

### Table 2: self-generated reference (Qwen3-8B, Qwen3.8-27B; GPU for 81 and scoring)

```
python scripts/80_build_self_reference_prompts.py --judge qwen3_8b --data-dir D --out G/prompts
python scripts/81_generate_self_reference.py --judge qwen3_8b --prompts G/prompts/self_reference_qwen3_8b.jsonl \
    --out G/generations/self_reference_qwen3_8b.jsonl
python scripts/82_build_self_reference_items.py --judge qwen3_8b --data-dir D --gen-dir G --out I
python scripts/42_score_local_judge.py --judge qwen3_8b --items I/self_reference_qwen3_8b_<dataset>.jsonl \
    --out S/<dataset>/self_reference/qwen3_8b.jsonl          # same for qwen3_8_27b
```
The judge writes the expected value from its own truncated Plain input (greedy, thinking off, 384 new tokens) and then
reads it as a reference. In the paper run each judge's items were scored in one run (batch 32).

### Table 4: reference x judge thinking (ComplexFuncBench, Qwen3.8-27B)

```
bash scripts/83_score_local_judge_thinking.sh I S <work dir>     # thinking on -> S/cfb/<input>/qwen3_8_27b_thinking.jsonl
python scripts/84_table4.py --data-dir D --scores-dir S --out-dir T   # thinking-off cells use S/cfb/<input>/qwen3_8_27b.jsonl
```
Thinking on: two-stage scoring with a thinking budget of 8,192 tokens (two shards, ~50 min each on one B200).

### Figure 1

```
python scripts/85_figure1_case.py --data-dir D --scores-dir S --deref-items I/deref_cfb.jsonl
```
Selection rule: a correct ComplexFuncBench history-only value whose DeRef reference names the gold value, not in the
Plain input, rejected by all four judges with Plain input at their tau and accepted with DeRef; shortest Plain input first.

### Figure 2a: FCRB-Relocate

```
# controlled set (shipped as D/data/fcrb_relocate.jsonl; steps 60-62 rebuild it)
python scripts/60_relocate_rewrite_prompts.py --data-dir D --out R/rewrite_prompts.jsonl
python scripts/61_relocate_rewrite_generate.py --prompts R/rewrite_prompts.jsonl --out R/rewrite_generations.jsonl   # GPU, Qwen3.8-27B
python scripts/62_relocate_build.py --data-dir D --export W/states/export.jsonl \
    --prompts R/rewrite_prompts.jsonl --generations R/rewrite_generations.jsonl --out D/data/fcrb_relocate.jsonl
# judge inputs and DeRef references
python scripts/63_relocate_inputs.py plain      --data-dir D --out I
python scripts/63_relocate_inputs.py plain_full --data-dir D --out I
python scripts/63_relocate_inputs.py prompts    --data-dir D --gen-dir G
python scripts/22_generate.py --prompts G/prompts/deref_fcrb_relocate.jsonl --out G/generations/deref_fcrb_relocate.jsonl --mode thinking --max-new 8192
python scripts/63_relocate_inputs.py deref      --data-dir D --gen-dir G --out I
# score into S/fcrb_relocate/{plain,plain_full,deref}/<judge>.jsonl (CLM has no full-history input)
python scripts/64_figure2a.py --data-dir D --scores-dir S --out-dir T
```
Step 62 needs the state export of step 07 (`W/states/export.jsonl`): the distractor values come from all
FC-RewardBench states, including states outside the analysis set.

### Figure 2b: correct FC-RewardBench defaults

```
python scripts/70_build_schema_facts.py --data-dir D --out I
python scripts/13_build_reference_items.py --method deref --data-dir D --gen-dir G --out I --datasets fcrb
# score I/schema_facts_fcrb.jsonl into S/fcrb/schema_facts/<judge>.jsonl (plain and deref as for Table 2)
python scripts/71_figure2b.py --data-dir D --inputs-dir I --scores-dir S --out-dir T
```
The n = 142 subset: correct default arguments whose DeRef reference contains the word "default" (case-insensitive,
Markdown emphasis removed).

### Figure 3: reference swap

```
python scripts/13_build_reference_items.py --method deref --data-dir D --gen-dir G --out I
python scripts/72_build_swap.py --data-dir D --inputs-dir I --out I
# score I/deref_swap_<dataset>.jsonl into S/<dataset>/deref_swap/<judge>.jsonl
python scripts/73_figure3.py --data-dir D --inputs-dir I --scores-dir S --out-dir T
```
Swap item ids are `<argument item_id>|deref_swap|<unit>` (an argument can be the wrong value of two units).

## Package layout

```
deref/build      sources -> data package, FCRB-Relocate           scripts 01-09, 60-62
deref/inputs     data package -> judge inputs                      scripts 10, 12-14, 63, 70, 72, 82
deref/generate   prompts and vLLM generation                       scripts 20, 22, 80, 81
deref/score      CLM, local judges (thinking off / on), Jev        scripts 40-47, 83 (+ .sh examples)
deref/stats      statistics and tables                             scripts 50-54, 64, 71, 73, 84-86
```

## Citation

```
@inproceedings{anonymous2026deref,
  title  = {Judges Follow the Reference: The Derivation Gap in Verifying Tool-Call Arguments},
  author = {Anonymous},
  year   = {2026}
}
```
