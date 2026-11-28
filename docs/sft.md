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
| `val_every` | 500 | masked loss on the first 128 `eval.jsonl` rows (all `math_reasoning`) at r = 1, 2, 4, 8 |
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

## The English mix run

`configs/sft-mix.toml` runs the same recipe on the larger English mix from [data.md](data.md#sft): FABLE, MiniMind's English rows, and smol-smoltalk, all through the answer-based English check. It starts again from the pretrain checkpoint and writes to `runs/sft-mix`, so `runs/sft` and `runs/sft-refine` stay as they are. It runs one epoch, because the mix is several times larger.

```bash
runs/sft-mix-supervisor.sh   # resumes up to 5 times, then evaluates into runs/eval/sft-mix.json
```

The supervisor script is local, like `runs/sft-supervisor.sh`. After training, it evaluates on the same 2,000 held-out rows as `runs/eval/sft.json`: masked loss at r = 1, 2, 4, 8, 16, and greedy answers at r = 4 and 8.

### Results

On 2026-11-27 and 28, from the same pretrain checkpoint, the run took 11 h 54 min of training in two sittings:
- 25,500 steps until a planned pause at 16:38;
- the remaining 18,243 steps after a reboot, resumed from checkpoint 25,500 with no steps lost.

CUDA failed after the laptop woke from suspend (`CUDA_ERROR_UNKNOWN`), and the reboot fixed it. `runs/sft-mix-pause.sh` and `runs/sft-mix-resume.sh` handled the pause and the restart. They are local, like the supervisor.

The run covered all 43,743 steps in CUDA f32. It averaged 8,480 padded tokens/s (MFU 1.8%) and a mean sampled `r` of 4.01. It covered 371.0M padded tokens and 255.0M supervised targets.

| Step | r = 1 | r = 2 | r = 4 | r = 8 |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 3.520 | 3.404 | 3.415 | 3.523 |
| 5,000 | 2.342 | 2.189 | 2.150 | 2.256 |
| 10,000 | 2.042 | 1.905 | 1.880 | 1.951 |
| 20,000 | 1.841 | 1.703 | 1.670 | 1.719 |
| 30,000 | 1.740 | 1.612 | 1.583 | 1.634 |
| 43,743 | **1.688** | **1.560** | **1.530** | **1.577** |

Validation masked loss on the first 128 eval rows. Those rows are all `math_reasoning`, so this curve measures math only. On it the mix ends below `runs/sft` (1.556 at r = 4) and just above `runs/sft-refine` (1.520).

The full eval covers all 2,000 rows, 500 per config, and the answers on 148 of them. There the mix is worse:

| | `runs/sft` | `runs/sft-refine` | `runs/sft-mix` |
| --- | ---: | ---: | ---: |
| loss, r = 4 | 1.037 | **1.008** | 1.174 |
| loss, r = 8 | 1.083 | **1.051** | 1.232 |
| macro accuracy, r = 8 greedy | 13.0% | **16.1%** | 11.0% |
| math | 2/50 | 1/50 | 2/50 |
| logic | 1/30 | 2/30 | 2/30 |
| science | 8/24 | 9/24 | 8/24 |
| JSON exact | 5/44 | **8/44** | 0/44 |
| JSON valid | 82% | 82% | 58% |

- The loss is higher at every `r`. The 128 math rows improved, so most of the rise is likely in the other three configs.
- JSON fell the most: exact matches went from 8 to 0 of 44, and valid JSON from 82% to 58%. FABLE is now 15% of the tokens, and smol-smoltalk answers in prose. Asked for a JSON object for Ana, who is 30, the model replies with a sentence about writing one.
- Answers at r = 8 beat r = 4 (11.0% against 8.4%), while the loss is lowest at r = 4. Both match the earlier runs.
- In the chat smoke prompts, the model now names Paris as the capital of France, where `runs/sft-refine` answered "The capital of France is the capital of France." Greedy answers still fall into loops that repeat one phrase, and the arithmetic is still wrong (12 + 7 = 12).

The keep rule for this run was macro accuracy at r = 8 greedy of at least `runs/sft-refine`'s 16.1%. The mix reached 11.0%, so `runs/best.json` stays on `runs/sft-refine`. The checkpoint stays in `runs/sft-mix` and can be chosen on the chat page.

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

## Chat in the browser

```bash
cargo build --release -p loopmind --features cuda
./target/release/loopmind serve --device cuda          # then open http://127.0.0.1:8080
```

`serve` runs a local chat page on this repo's inference code. Nothing goes to an outside service. It offers every run directory under `runs/` that has a `latest.json`, and it starts on the checkpoint, loop count, and sampling in `runs/best.json`. You can change these on the page:

- the checkpoint;
- the loop count `r`;
- the temperature, or a preset: greedy, `t0.3-k20`, or `t0.7-k40`;
- top-k, the repetition penalty, max tokens, and the seed;
- a system message.

When `runs/eval/<run>.json` exists, the page shows that checkpoint's eval accuracy and its held-out loss at the chosen `r`. The text streams token by token. Esc or Stop ends an answer early, and the server stops generating for it. Settings are kept in the browser's local storage.

**Send earlier turns** is off by default, so each message is answered on its own. SFT is mostly single-turn: 4.7% of train rows are multi-turn dialogues, and all 2,000 held-out rows have one user turn. With an earlier exchange in context, the model tends to copy its previous answer. Turn the setting on to test multi-turn chat. When it is on, the oldest turns are dropped, as in `chat`, once the prompt plus max tokens would pass the 1024-token context.

On the 4070, with no KV cache, answers stream at about 24–35 tokens/s at r = 4 or 8, once the first answer has compiled the kernels. The server listens on 127.0.0.1 and refuses requests whose `Host` is not a local name. `--host 0.0.0.0` opens it to the network and turns that check off. One worker thread holds one checkpoint and answers one request at a time. Switching checkpoints reloads the weights, which takes about a second.

| Endpoint | |
| --- | --- |
| `GET /` | the page |
| `GET /api/info` | checkpoints, their eval summaries, and the starting settings |
| `POST /api/chat` | `{checkpoint, messages: [{role, content}], system, loops, temperature, top_k, repetition_penalty, max_new, seed}`. Streams NDJSON: one `start` event, then a `text` event per token with the full text so far, then `done` with the final text, token count, speed, and why it stopped (`end`, `length`, or `cancelled`). Errors are an `error` event. |
