# Eval, loop-dial tuning, and refinement

After SFT, `scripts/refine.sh` evaluates the model, tunes the loop count and sampling, runs a low-learning-rate SFT pass and self-generated DPO, and keeps whichever checkpoint wins. It writes `runs/best.json`, which `loopmind chat` uses when no `--checkpoint` is given, and `runs/eval/summary.md`.

```bash
DEVICE=cuda scripts/refine.sh                  # after runs/sft has finished
WAIT_PID=<sft pid> DEVICE=cuda scripts/refine.sh   # start when that process exits
```

Every stage leaves `runs/refine/<stage>.done`, so running the script again resumes. Training stages retry with `--resume` up to three times. A failed stage is logged in `runs/refine/pipeline.log` and the pipeline continues with the checkpoints it has.

## Eval

```bash
loopmind eval --checkpoint runs/sft --out runs/eval/sft.json --device cuda
```

The rows are `data/prepared/sft/eval.jsonl`: 2,000 checkable rows that never enter training, 500 from each of `math_reasoning`, `logic_reasoning`, `science_knowledge`, and `structured_output`.

1. Masked loss on all 2,000 rows at r = 1, 2, 4, 8, 16 (`--nll-loops`). It is the same token-mean NLL as SFT validation.
2. Greedy answers for the first 50 rows of each config at r = 1, 2, 4, 8, 16 (`--gen-loops`, `--gen-rows`), at most 160 new tokens.
3. At the best greedy `r`, the sampled presets `t0.3-k20` and `t0.7-k40` (`--sampling`). Every preset uses repetition penalty 1.1.
4. The six chat smoke prompts in `eval::SMOKE_PROMPTS`, at the recommended setting.

| Config | What is compared | Scorable eval rows |
| --- | --- | ---: |
| `math_reasoning` | the final number (`Final answer:`, `The answer is:`, `\boxed{}`, else the last number) | 500 |
| `logic_reasoning` | the text after the last `Answer:`, case and whitespace ignored | 296 |
| `science_knowledge` | the option letter, for multiple-choice references like `B. deep ocean` | 231 |
| `structured_output` | the first JSON value, which must equal the reference's; validity is reported too | 459 |

Rows whose reference has nothing to compare are unscored, not wrong. `references_score_against_themselves` checks every reference against itself. The macro accuracy is the mean over configs with scored rows. The recommendation is the highest macro accuracy; ties go to the lower masked loss at that `r`, then to greedy. If every setting scores zero, it is greedy at the `r` with the lowest masked loss.

Generation has no KV cache. Prompts of the same length share a batch, which needs no padding mask because every row grows by one token per step. The JSON report goes to `--out`, and a Markdown version with sample answers goes next to it.

## Refinement pass

`configs/sft-refine.toml` runs one more epoch over the SFT train split from the final SFT checkpoint, at a tenth of the SFT learning rate (2e-5 to 2e-6). The pipeline keeps it only if its held-out masked loss, at the tuned `r`, is lower than the SFT model's.

## Self-generated DPO

```bash
loopmind dpo-pairs --checkpoint runs/sft --out runs/dpo/pairs.jsonl --device cuda --loops 4
loopmind dpo --config configs/dpo.toml --init runs/sft --device cuda
```

A pair is a training prompt from the SFT shards with two answers. The chosen one is the dataset's reference answer. The rejected one is the model's own answer, sampled at `t0.7-k40` and closed with `<|im_end|>\n<|eos|>` when it stopped on its own. Only prompts whose reference answer fits in `--max-new` (160) tokens are used, so both sides had the same budget. Prompts are drawn from the lengths with the most prompts, so batches of 32 need no padding, and at most 64 prompts come from any one length.

`loopmind dpo` minimizes `-log sigmoid(beta * ((log pi_c - log ref_c) - (log pi_r - log ref_r)))` over the masked answer tokens (Rafailov et al., 2023). The reference is a frozen copy of the starting weights. A pair's two rows and the reference share one sampled loop count, so `r` cancels out of the ratio. `nll_weight` adds the chosen answer's token-mean NLL, which keeps the policy close to text it already writes.

| Knob | Value |
| --- | ---: |
| pairs | 4,096, the first 256 held out |
| `beta` | 0.1 |
| `nll_weight` | 0.2 |
| `learning_rate` | 1e-5 to 1e-6, warmup 20 |
| `pairs_per_step` | 8 |
| `epochs` | 2 |

Validation reports the held-out pairs' DPO loss, mean margin, and the fraction with a positive margin, at r = 4. The pipeline keeps the DPO checkpoint only if macro answer accuracy rises by at least 0.5 points and the held-out masked loss rises by no more than 0.05.

## Chat defaults

`runs/best.json` holds the winning checkpoint, the tuned `r`, and the sampling preset. `loopmind chat` without `--checkpoint` reads it. `--loops`, `--temperature`, `--top-k`, and `--repetition-penalty` still override it.

## The 4070 run

The pipeline ran on 2026-11-27, starting when SFT finished. It took 4 h 42 min, with no failed stages or retries.

| Stage | Start | Time |
| --- | ---: | ---: |
| eval of `runs/sft` (full sweep) | 03:54 | 1 h 31 min |
| refinement pass, 9,595 steps | 05:26 | 1 h 46 min |
| eval of `runs/sft-refine` | 07:12 | 18 min |
| DPO pairs | 07:30 | 43 min |
| DPO, 960 steps | 08:12 | 9 min |
| eval of `runs/dpo` | 08:21 | 15 min |

### Loop dial

On the SFT model. Loss is over all 2,000 eval rows. Answers are greedy, on 50 rows per config: 148 scored, of which 50 math, 30 logic, 24 science, and 44 JSON.

| r | Held-out loss | Greedy macro | Correct | Answer time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.158 | 12.0% | 14 | 162 s |
| 2 | 1.057 | 10.2% | 11 | 263 s |
| 4 | **1.037** | 12.7% | 16 | 502 s |
| 8 | 1.083 | **13.0%** | 16 | 896 s |
| 16 | 1.356 | 12.1% | 13 | 1,689 s |

At r = 8, `t0.3-k20` scored 11.9% and `t0.7-k40` scored 9.8%.

- Loss is lowest at r = 4, as in SFT validation. It is worse at r = 16 than at r = 1: the model does not gain from passes beyond the depths it was trained on.
- Accuracy hardly moves with `r`. r = 8 won the recommendation with 13.0% against 12.7% at r = 4, but both settings got 16 answers right. r = 8 has one more science and one more JSON answer, r = 4 one more math and one more logic, and the macro average weighs those differently. The choice between them is noise. r = 4 has 0.046 lower loss and answers in 56% of the time. `loopmind chat --loops 4` overrides `runs/best.json`.
- Greedy beat both sampled presets.

### Checkpoints

At the tuned setting, r = 8 greedy, on the same 148 scored rows:

| Checkpoint | Held-out loss, r = 4 | Held-out loss, r = 8 | Macro | Correct | math | logic | science | JSON exact | JSON valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT | 1.037 | 1.083 | 13.0% | 16 | 2/50 | 1/30 | 8/24 | 5/44 | 82% |
| SFT + refinement pass | **1.008** | **1.051** | **16.1%** | **20** | 1/50 | 2/30 | 9/24 | 8/44 | 82% |
| DPO from the refinement pass | 1.129 | 1.203 | 9.7% | 10 | 1/50 | 1/30 | 8/24 | 0/44 | 78% |

`runs/best.json` is the refinement pass.

- **Refinement pass: kept.** It lowered held-out loss at every `r`, by 0.029 to 0.044 nats. That gain is consistent. The accuracy gain is four answers, mostly JSON.
- **DPO: discarded.** Held-out loss rose 0.15 nats and exact JSON matches fell to 0. The details are below.
- Math (1–2 of 50) and logic (1–2 of 30) are near zero for every checkpoint. Science is 8–9 of 24. The scored references are spread over five letters, with D the most common at 8, so this is little above guessing. JSON is the one config with a real signal. The task is pulling the fields of a prompt into JSON: the model usually gets the shape and most fields right, and the misses are a wrong value or a repeated record.

### DPO

`dpo-pairs` drew 100,279 candidate prompts and wrote 4,096 pairs. 3,257 of the model's answers ended on their own. 107 were identical to the reference and were skipped. Prompts averaged 56 tokens, chosen answers 65, and rejected 81.

| Steps | Chosen reward | Rejected reward | Margin | Train pair accuracy |
| --- | ---: | ---: | ---: | ---: |
| 1–100 | −0.50 | −1.64 | 1.14 | 0.78 |
| 101–480 | −0.95 | −3.06 | 2.11 | 0.86 |
| 481–960 | −0.77 | −3.71 | 2.94 | 0.94 |

A reward is `beta * (log pi - log ref)` summed over an answer.

- The held-out pairs were easy. Preference accuracy reached 0.82 by step 100 and ended at 0.84, while the margin grew from 1.08 to 1.76.
- The loss was met by lowering both answers, the model's own samples faster than the references. A chosen reward of −0.8 at `beta` 0.1 is about 8 nats less log-probability per reference answer, or about 0.12 per token. That matches the 0.15-nat rise in held-out loss. The `nll_weight` 0.2 term did not hold the chosen answers in place. Falling chosen likelihood is a known DPO failure (Pal et al., 2024; Razin et al., 2024).
- Changes to try next: a larger `nll_weight` (1.0, as in iterative RPO), a lower learning rate, or a few hundred steps instead of 960, since held-out accuracy was 0.82 at step 100 and never passed 0.86. The pairs in `runs/dpo/pairs.jsonl` can be reused.

### Chat

A 10M model gives fluent fragments, not answers. Two of the six smoke prompts, with the refinement-pass model at r = 8 greedy:

```text
> What is the capital of France?
The capital of France is the capital of France.

> List three colors.
1. Audo: Audo is a type of colors.
2. Audo is a type of colors.
3. Audo is a type of colors.
```

All six, for each checkpoint, are in `runs/eval/<checkpoint>.md`.
