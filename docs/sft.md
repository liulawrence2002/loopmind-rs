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
- Pad each row with `<|pad|>` to a multiple of `pad_multiple` (16), about 2% extra tokens. Rows are not packed into a shared width: each row runs alone with its own sampled loop count, as a pretrain sequence does, so a shared width would only add padding.
- The loss is next-token cross-entropy on supervised targets only. The M2 mask is 1 on assistant content, the `<|im_end|>` that closes it, and the final `<|eos|>`. Token `t` is a target when `mask[t]` is 1, and it is predicted from position `t - 1`.
- A step is the token-mean over every supervised target in its 16 rows: each row's masked NLL sum is divided by the step's target count before backward, and the gradients are summed, not averaged. The load-balancing term is averaged over rows. A long answer weighs more than a short one.
- Loop counts, truncated backprop, AdamW, the parameter groups, and clipping are the pretrain ones. The optimizer starts fresh; only the weights come from `init`.

The train split has 51.0M tokens (mean 332 per row, median 269) and 22.75M supervised targets. Padding to 16 makes 52.2M.

| Knob | Value | Why |
| --- | ---: | --- |
| `rows_per_step` | 16 | about 5,400 padded tokens and 2,400 targets per step |
| `epochs` | 2 | 19,190 steps |
| `learning_rate` | 1e-4 | a third of the pretrain peak; cosine floor 1e-5, warmup 100 |
| `weight_decay` | 0.1 | as pretrain |
| `val_every` | 500 | masked loss on the first 128 `eval.jsonl` rows at r = 1, 2, 4, 8 |
| `checkpoint.every` | 500 | keep 3 |
| `samples` | 4 | greedy answers at r = 4 to the first eval prompts, printed at the end and written to `runs/sft/samples.txt` |

Validation uses held-out rows: `eval.jsonl` is 2,000 checkable rows that are not in the train shards. The validation loss is the same token-mean masked NLL. It is not an accuracy.

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
