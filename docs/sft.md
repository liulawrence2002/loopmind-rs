# Supervised finetuning and chat

`loopmind sft` finetunes a pretrain checkpoint on the English chat split from [data.md](data.md). `loopmind chat` talks to the result.

```bash
cargo run -p loopmind --release -- prepare-sft --config configs/data.toml
cargo run -p loopmind --release --features cuda -- sft --config configs/sft.toml --device cuda
cargo run -p loopmind --release --features cuda -- chat --checkpoint runs/sft --device cuda
```

`init` in `configs/sft.toml` is `runs/pretrain`. A run directory resolves through its `latest.json`, so SFT starts from the last pretrain checkpoint. `--init` points somewhere else. `--resume`, `--max-steps`, and `--device` work as they do for `pretrain`. Logs go to stdout and `runs/sft/metrics.jsonl`. Checkpoints use the pretrain layout under `runs/sft/checkpoints/`.

## What a step does

- Take the next `rows_per_step` (16) conversations from a seeded shuffle of the 153,517 train rows. One epoch visits every row once. The cursor is in the checkpoint.
- Pad each row with `<|pad|>` to a multiple of `pad_multiple` (64). Rows are not packed into a shared width: each row runs alone with its own sampled loop count, as a pretrain sequence does, so a shared width would only add padding.
- The loss is next-token cross-entropy on supervised targets only. The M2 mask is 1 on assistant content, the `<|im_end|>` that closes it, and the final `<|eos|>`. Token `t` is a target when `mask[t]` is 1, and it is predicted from position `t - 1`.
- A step is the token-mean over every supervised target in its 16 rows: each row's masked NLL sum is divided by the step's target count before backward, and the gradients are summed, not averaged. The load-balancing term is averaged over rows. A long answer weighs more than a short one.
- Loop counts, truncated backprop, AdamW, the parameter groups, and clipping are the pretrain ones. The optimizer starts fresh; only the weights come from `init`.

The train split has 51.0M tokens (mean 332 per row, median 269) and 22.75M supervised targets.

| `pad_multiple` | Padded tokens | Distinct widths | 4070 probe |
| ---: | ---: | ---: | ---: |
| 16 | 52.2M (+2.3%) | 63 | 0.86 s/step |
| 64 | 56.0M (+9.6%) | 16 | 0.62 s/step |
| 128 | 60.7M (+19.0%) | 8 | |

Each new width costs CubeCL kernel compiles and an autotune pass, so 64 was faster than 16 despite the extra padding. The probes were 30–40 steps from the final pretrain checkpoint.

| Knob | Value | Why |
| --- | ---: | --- |
| `rows_per_step` | 16 | about 5,800 padded tokens and 2,400 targets per step |
| `epochs` | 2 | 19,190 steps |
| `learning_rate` | 1e-4 | a third of the pretrain peak; cosine floor 1e-5, warmup 100 |
| `weight_decay` | 0.1 | as pretrain |
| `val_every` | 500 | masked loss on the first 128 `eval.jsonl` rows at r = 1, 2, 4, 8 |
| `checkpoint.every` | 500 | keep 3 |
| `samples` | 4 | greedy answers at r = 4 to the first eval prompts, printed at the end and written to `runs/sft/samples.txt` |

Validation uses held-out rows: `eval.jsonl` is 2,000 checkable rows that are not in the train shards. The validation loss is the same token-mean masked NLL. It is not an accuracy.

## The 4070 run

On 2026-11-27, from the final pretrain checkpoint (step 6,000), the run took 3 h 32 min: all 19,190 steps in one attempt, with no restarts. It used CUDA f32. It averaged 9,050 padded tokens/s (MFU 1.7%) and a mean sampled `r` of 4.02. It covered 111.9M padded tokens and 45.5M supervised targets.

| Step | Epoch | r = 1 | r = 2 | r = 4 | r = 8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 1 | 3.340 | 3.219 | 3.231 | 3.337 |
| 2,000 | 1 | 2.632 | 2.474 | 2.458 | 2.599 |
| 5,000 | 1 | 2.137 | 1.987 | 1.956 | 2.051 |
| 9,500 | 1 | 1.867 | 1.729 | 1.696 | 1.771 |
| 12,000 | 2 | 1.808 | 1.673 | 1.640 | 1.698 |
| 15,000 | 2 | 1.752 | 1.620 | 1.589 | 1.643 |
| 19,190 | 2 | **1.714** | **1.587** | **1.556** | **1.608** |

Validation masked loss on the first 128 eval rows. The full curve is in `runs/sft/metrics.jsonl`.

- From step 2,000 on, r = 4 is the best of the four. Its lead over r = 1 peaked at 0.20 nats around step 4,500 and ended at 0.16. r = 8 is worse than r = 4 at every point, Training draws `r` from a clamped Poisson with mean 4, so only 5.1% of training rows ran 8 or more passes. The loop dial is tuned after training ([refine.md](refine.md)).
- The second epoch took r = 4 from 1.696 to 1.556. It was still falling at the end, at about 0.004 per 500 steps, as the cosine schedule neared its 1e-5 floor.
- The mean train loss over the last 500 steps was 1.81. Train loss is noisier than validation, because a step's 16 rows vary in length and config.

`runs/sft/samples.txt` holds greedy answers at r = 4 to the first four eval prompts, all grade-school math, with no repetition penalty. The model has learned the answer format, such as `Let's denote ... as \( x \).` and `Final answer:`. Its arithmetic is wrong, and three of the four answers fall into a loop that repeats one phrase:

```text
He spent 4*4=12 hours a day
So he spent 12+12=8 hours a day

Final answer: 8
```

The reference answer is 12. The `chat` defaults add a repetition penalty of 1.1 and sampling for this reason.

## Chat

```bash
loopmind chat --checkpoint runs/sft --device cuda
loopmind chat --checkpoint runs/sft --prompt "What is the capital of France?"
```

Without `--prompt`, `chat` reads one user turn per line from stdin. `/reset` clears the history; `/exit` or Ctrl-D quits. Each turn is rendered with the M2 template and a generation prompt (`<|im_start|><|assistant|>\n`). Generation stops at `<|im_end|>` or `<|eos|>`. When the history plus `--max-new` would pass the 1024-token context, the oldest non-system turns are dropped.

| Flag | Default | |
| --- | ---: | --- |
| `--loops` | 4 | the loop dial `r` |
| `--temperature` | 0.7 | 0 is greedy |
| `--top-k` | 40 | 0 keeps the whole vocabulary |
| `--repetition-penalty` | 1.1 | divides positive logits (multiplies negative ones) of ids already in the context |
| `--max-new` | 256 | per answer |
| `--system` | none | optional system turn |

There is no KV cache. Each new token reruns the window, which at r = 4 is fine for a 10M model.
