# Pretraining

`loopmind pretrain` trains the default 10M model on the packed shards from [data.md](data.md). One optimizer step is AdamW on `tokens_per_step` tokens. The per-sequence loop count is a clamped Poisson with mean 4 and max 16, and only the last 4 passes receive gradients. The loss is next-token cross-entropy plus the Switch load-balancing term already scaled by `aux_loss_coef`.

```bash
cargo run -p loopmind --release --features cuda -- pretrain --config configs/pretrain.toml --device cuda
```

Stop early with `--max-steps N`. Resume with `--resume`; it loads `runs/pretrain/latest.json` and continues from the next step. Logs go to stdout and `runs/pretrain/metrics.jsonl`.

## What a step does

- Draw the next packed sequences in a seeded shuffle, so one epoch walks the ~50M tokens once. A remainder smaller than one step is spilled into the following epoch.
- Run each sequence on its own loop count. Sequences in a microbatch do not share `r`.
- Shift the logits by one position. Token `t` is predicted from the prefix that ends at `t - 1`.
- Accumulate `tokens_per_step / (micro_batch * seq_len)` microbatches, average the gradients, clip the global L2 norm to `grad_clip`, then take one AdamW step.
- Weight decay hits the matrices and skips the token embedding and every RMSNorm scale (`embed` and `norm` in the parameter path).
- The learning rate warms up linearly for `warmup_steps`, then cosine-decays from `learning_rate` to `min_learning_rate` across the planned epoch.

Checkpoints land in `runs/pretrain/checkpoints/step-NNNNNNNN/` and contain `model.bpk`, `optim.bpk`, and `state.json` (step, data cursor, loop RNG, schedule length, model config). `checkpoint.keep` is how many of those directories to retain. The run also writes one at the end.

## Device and precision

`--device cpu` uses Burn's flex backend in f32. That is what CI uses.

`--device cuda` needs a binary built with `--features cuda`. On CUDA the run tries a bf16 matmul, then f16, and stays on f32 if those values are not finite. The weights are stored in the dtype that worked. Burn updates those tensors in place, so this is not a separate f32 master-weight mixed-precision optimizer. `--device wgpu` needs `--features wgpu` and stays on f32.

There is no GPU on the machine that produced the smoke numbers below, so that probe was not executed here.

## Defaults for the ThinkPad

`configs/pretrain.toml` is aimed at an RTX 4070 Laptop with 8 GB:

| Knob | Value | Why |
| --- | ---: | --- |
| sequence length | 1024 | the packed shards |
| `micro_batch` | 1 | one sequence in the autograd graph at a time |
| `tokens_per_step` | 8192 | 8 accumulated sequences |
| `epochs` | 1 | about 6,104 optimizer steps over 48,826 sequences |
| `learning_rate` | 3e-4 | cosine floor 3e-5, warmup 100 steps |
| `grad_clip` | 1.0 | global L2 |
| `val_every` | 200 | 16 val sequences at r = 1, 2, 4, and 8 |
| `peak_tflops` | 15 | FP32 peak used for the MFU line |

The active checkpoint is about 10M parameters. Activations dominate. The core is re-routed every pass and every expert is evaluated, so a microbatch of 1 at 1024 tokens is the comfortable setting; raise `micro_batch` only after watching memory.

A fresh uniform model should start near `ln(6400)` ≈ 8.76 nats. A run that is learning leaves the high 8s during warmup and is in the 4s–6s by the end of this epoch. Validation at r = 1, 2, 4, 8 is the loop-depth dial: after pretraining, a higher `r` should not be wildly worse, and r = 4 is the training mean.

## Smoke run

This VM has no GPU. A release `--device cpu` run of `configs/pretrain.toml` got through **199 optimizer steps** (1,630,208 packed tokens) in about 88 minutes before the process was killed. Median throughput was **308 tokens/s** (mean 314, range 240–431). `mfu` sat at about **0.06%** of the 15 TFLOP/s figure in the config, which is the CPU doing the whole unfused expert loop.

| Step | Loss | Aux | LR | Grad norm | Mean r | tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.997 | 0.201 | 3.0e-6 | 1.62 | 4.62 | 247 |
| 30 | 8.515 | 0.132 | 9.0e-5 | 1.48 | 4.38 | 305 |
| 100 | 7.140 | 0.069 | 3.0e-4 | 0.39 | 3.25 | 351 |
| 150 | 6.752 | 0.099 | 3.0e-4 | 0.41 | 4.62 | 283 |
| 199 | 6.334 | 0.082 | 3.0e-4 | 0.48 | 4.00 | 326 |

`ln(6400)` is 8.76. The run started a bit above that and was in the low 6s when it stopped, still inside the 100-step warmup's tail and the first part of the cosine. A greedy sample after step 1 (`--loops 4`, prompt "Once upon a time") repeated a few words. That is what a model at loss ~9 does.

The kill was an out-of-memory SIGKILL at about 15 GB. Each microbatch's autograd tape stayed alive because the accumulated gradients still pointed at it, so RSS climbed by tens of MB every step. Gradients are now detached before they are accumulated. A follow-up 12-step run on the same shards logged the same early losses (step 1 was 8.997 again) and, while a step was in flight, sat around 4.0–4.7 GB instead of climbing without bound. One 1024-token backward is large; it fits in 8 GB at `micro_batch = 1`.

## 4070 estimate

There is no CUDA device here, so this is an extrapolation. The CPU does about 300 tokens/s. The same step is a long chain of small expert kernels, so the laptop 4070 will not hit the 15 TFLOP/s peak; the logged MFU is there to show that. A 15–40× speedup over this CPU is the plausible band for that unfused loop, about **5,000–12,000 tokens/s**. The epoch is 48,826 sequences, 6,104 optimizer steps, ~50M tokens:

- 5,000 tok/s → about 3 hours
- 12,000 tok/s → about 70 minutes

Plan on **1–3 hours** for the epoch. The first CUDA log line should be a few hundred milliseconds to a couple of seconds per step, not the ~26 s seen here, and `nvidia-smi` memory should stay under 8 GB. If a step is still many seconds, the expert loop is launch-bound and the run is still correct, just slower than the optimistic end of that range.

## Greedy samples

```bash
cargo run -p loopmind --release -- sample \
  --checkpoint runs/pretrain/checkpoints/step-00000200 \
  --prompt "Once upon a time" \
  --loops 4 \
  --max-new 64
```

`--loops` is a fixed `r`. The checkpoint directory is the one that contains `model.bpk`.

## Running it on the ThinkPad P1

Prepare is CPU-only and is documented in [data.md](data.md). Pretraining is the GPU job. WSL2 is the path that matches this repo.

1. Install the current NVIDIA driver on Windows. WSL2 uses that driver; do not install a second Linux driver inside the distro.
2. In Ubuntu on WSL2: `sudo apt update && sudo apt install -y build-essential python3-pip`. Install the CUDA toolkit that matches the driver (the `nvcc` package from Ubuntu or NVIDIA's WSL CUDA repo). `nvcc --version` should print.
3. Install Rustup. The repo pins stable via `rust-toolchain.toml` (1.95 or newer).
4. From the repo root, with the shards already built:

```bash
cargo run -p loopmind --release --features cuda -- pretrain \
  --config configs/pretrain.toml \
  --device cuda
```

The first CUDA build compiles CubeCL and takes a while. Later runs reuse it. Expect a few GB under `runs/pretrain/` for two checkpoints plus the jsonl log. The 4070 Laptop has 8 GB; if the process is killed, drop `micro_batch` (it is already 1) or the sequence length is fixed by the shards, so the next lever is not to raise `micro_batch`.

Resume after a reboot with the same command plus `--resume`.

Watch the first validation block. `val r1`, `r2`, `r4`, and `r8` are the loss at those fixed loop counts. `mfu` divides estimated training FLOPs by `peak_tflops` (15). On this model the expert loop is a sequence of small kernels, so the 4070 will sit well below that peak; tokens/sec is the number to trust.

To eyeball text:

```bash
cargo run -p loopmind --release --features cuda -- sample \
  --checkpoint runs/pretrain/checkpoints/step-00002000 \
  --prompt "The capital of France is" \
  --loops 4 \
  --device cuda
```
