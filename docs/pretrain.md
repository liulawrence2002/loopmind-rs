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

Checkpoints land in `runs/pretrain/checkpoints/step-NNNNNNNN/` and contain `model.bpk`, `optim.bpk`, and `state.json` (step, data cursor, loop RNG, schedule length, model config). `checkpoint.keep` is how many of those directories to retain. The run also writes one at the end. A step's checkpoint is written before that step's validation, so a failed validation never costs the checkpoint.

## Device and precision

`--device cpu` uses Burn's flex backend in f32. That is what CI uses.

`--device cuda` needs a binary built with `--features cuda`. `--device wgpu` needs `--features wgpu`. Both train in f32 by default.

`precision` in `configs/pretrain.toml` is `"f32"` (the default) or `"auto"`. With `"auto"`, a CUDA run tries a bf16 matmul, then f16, and casts every weight to the first dtype that returns finite values. Burn then updates those tensors in place, so there is no f32 master copy.

On the RTX 4070 Laptop, the first `"auto"` run picked bf16 and its first microbatch's loss was not finite. The cause was three f32 helper tensors meeting bf16 activations: the RoPE tables, the one-hot of the routed expert ids, and the slot fractions in the balance loss. Burn's flex backend panics on that mix (`dtype mismatch (expected BF16, got F32)`). On CUDA it read as garbage. All three are now cast to the activations' dtype, and `bf16_weights_give_a_finite_loss_and_gradient` runs a bf16 forward and backward on flex. f32 stays the default because bf16 without a master copy is still a poor trade on this model:

- An RMSNorm gain starts at 1.0, where half a bf16 ulp is about 0.004. An AdamW step is about `lr`, at most 3e-4, so the gains would never move.
- Decoupled weight decay at `lr * 0.1` scales each weight by about `1 - 3e-5`, which rounds back to 1 in bf16.
- Near the end of the cosine (`lr` 3e-5), most updates to the matrices are below half an ulp as well.

The model is 10M parameters, so f32 weights plus AdamW state are about 160 MB. Memory goes to activations, not weights. The run is launch-bound (below), so f32 costs little throughput.

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
| `precision` | f32 | see above |
| `peak_tflops` | 15 | FP32 peak used for the MFU line |

The active checkpoint is about 10M parameters. Activations dominate. The core is re-routed every pass and every expert is evaluated, so a microbatch of 1 at 1024 tokens is the comfortable setting; raise `micro_batch` only after watching memory.

A fresh uniform model should start near `ln(6400)` ≈ 8.76 nats. A run that is learning leaves the high 8s during warmup and is in the 4s–6s by the end of this epoch. Validation at r = 1, 2, 4, 8 is the loop-depth dial: after pretraining, a higher `r` should not be wildly worse, and r = 4 is the training mean.

Validation and `sample` run on the model's non-autodiff copy (`Module::valid`). On the training device, an untruncated r = 8 forward would keep all eight passes' activations alive, twice the four backprop passes a training step holds. The first 4070 run did that and ran out of device memory in the step-200 validation, before the step-200 checkpoint was written. With `valid`, validating every step on the same card peaks at 5.6 GB.

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

## RTX 4070 Laptop

Measured on the dev box: RTX 4070 Laptop (8 GB, driver 550, CUDA 12.5 toolkit), release `--features cuda`, `configs/pretrain.toml` with `precision = "f32"`.

| | Value |
| --- | ---: |
| Throughput, first 144 steps | 4,091 tok/s mean, about 2.0 s per 8,192-token step |
| Range per step | about 3,400–5,600 tok/s; longer sampled `r` is slower |
| `nvidia-smi` memory | 6.7 GB by step 6, flat at 7.4 GB from step ~77 |
| GPU utilization | about 40% |
| `mfu` | about 0.85% of 15 TFLOP/s |
| Epoch | 6,104 steps, about 3.5 hours before the fused experts below |

The early curve matches the CPU smoke below: loss 7.146 at step 100 (CPU 7.140) and 6.765 at step 150 (CPU 6.752).

The GPU sat at ~40% because the step was launch-bound. Each core layer evaluated all 32 experts as separate `[1024, 256] × [256, 128]` SwiGLUs, about 100 small matmuls per layer per pass, plus a host read of the expert-slot fractions after every layer.

### Fused experts

`Mixture::forward` now concatenates the experts' weights into one `[256, 4096]` gate, one `[256, 4096]` up, and one `[4096, 256]` down. The router's gates are repeated across each expert's 128 columns, so `down_e(h_e) * g_e` summed over experts becomes `concat_e(h_e * g_e) @ concat_e(down_e)`: three large matmuls instead of 96 small ones. The core is shared across loop passes, so a forward concatenates once per layer and every pass reuses it. Gradients flow back through the concat into each expert's own `Linear`, so the checkpoint layout is unchanged.

The same change builds the RoPE tables and causal mask once per forward instead of once per attention call. It reads the expert-slot fractions back once per forward instead of once per layer per pass, the global grad norm in one read instead of one per parameter tensor, and the loss with its aux term in one read.

`fused_experts_match_the_per_expert_loop` checks outputs and every expert weight's gradient against the old loop to 1e-5. PyTorch parity on the tiny config is still 9.5e-7. On the real run, step 1,601 resumed from the step-1,600 checkpoint gave loss 5.0310 and grad norm 1.039 with the fused binary, against 5.0308 and 1.038 from the old one.

| From step 1,600 | Per-expert loop | Fused |
| --- | ---: | ---: |
| Throughput | ~4,100 tok/s (up to ~7,800 on an idle CPU) | ~12,000 tok/s (10,000–15,000) |
| GPU utilization | ~40% | ~75–85% |
| `nvidia-smi` memory | 7.4 GB | 6.7–7.6 GB (4.4 GB after the fix below) |
| `mfu` | ~0.85% | ~2.3% |

At that rate an epoch is about 70 minutes.

### Memory grew with `r`

At step 5,701 the run ran out of device memory on an ordinary training step. Passes before the last `bptt_passes` were detached, but autodiff still kept their activations alive until backward, so memory grew with the sampled loop count. With a loop mean of 14, a synthetic run failed on its first step. The live run sat at ~7.6 GB, so a rare high-`r` sequence was enough.

`recur` now runs those passes on the model's non-autodiff copy (`Module::valid`, `Tensor::inner`) and lifts the result back as a constant. Values and gradients are unchanged: those passes never received gradients. `passes_before_the_window_run_untracked_but_keep_the_values` checks the logits against the untruncated forward on an autodiff device.

| Peak `nvidia-smi` memory | Before | After |
| --- | ---: | ---: |
| Loop mean 14 (r ≈ 13) | out of memory | 4.4 GB |
| Loop mean 4, live run | 7.6 GB | 4.4 GB |
| Loop mean 2 | 3.3 GB | 3.3 GB |

Resumed from the step-5,600 checkpoint, steps 5,601–5,604 reproduced the crashed run's loss, grad norm, and loop counts to four decimals.

## The 500M run

Ready, not started. The data is built in `data/prepared-500m` (2026-11-28). The 50M run gave about 5 tokens per parameter. That is far below the roughly 20 per parameter usually taken as compute-optimal, and small models are now trained well past that. Its base model knows few facts: before SFT it could not finish "The capital of France is". SFT cannot add what pretraining missed. The 500M run gives about 50 tokens per parameter, or about 210 per active parameter.

```bash
./target/release/loopmind prepare-pretrain --config configs/data-500m.toml   # 11 min, CPU only (done)
setsid nohup scripts/pretrain-500m.sh > /dev/null 2>&1 &                     # about 11-12 h on the 4070
tail -f runs/pretrain-500m/train.log
```

`scripts/pretrain-500m.sh` resumes from the latest checkpoint when `runs/pretrain-500m/latest.json` exists, including on its first attempt. It trims `metrics.jsonl` back to that checkpoint and retries up to 5 times. To pause, kill the script first, then the trainer, and run the script again to continue. After a laptop suspend, CUDA can fail with `CUDA_ERROR_UNKNOWN` until `nvidia_uvm` is reloaded or the machine reboots. Stop the run before closing the lid.

| | 50M run | 500M run |
| --- | --- | --- |
| Data config | `configs/data.toml` | `configs/data-500m.toml` |
| Mix | FineWeb-Edu 50%, Cosmopedia 20%, TinyStories 20%, FABLE text 10% | FineWeb-Edu 53%, Cosmopedia 25%, Simple English Wikipedia 10%, TinyStories 10%, FABLE text 2% ([data.md](data.md#the-500m-mix)) |
| Tokens per step | 8,192 | 32,768 |
| Steps | 6,104 | 15,259 |
| Peak learning rate | 3e-4, floor 3e-5 | 6e-4, floor 6e-5 |
| Warmup | 100 steps | 300 steps (9.8M tokens) |
| Validation | every 200 steps, 16 sequences, r = 1, 2, 4, 8 | every 250 steps, 64 sequences, r = 1, 2, 4, 8, 16 |
| Checkpoints | every 200 steps, keep 2 | every 250 steps and at the end, all kept (62, about 7 GB) |

The model, the loop sampling (clamped Poisson, mean 4, max 16, gradients through the last 4 passes), f32, AdamW, weight decay, gradient clipping, context, and seed are unchanged.

- **Batch and learning rate.** Four times the batch means a quarter of the optimizer steps per token. The peak learning rate is doubled, the square-root rule for a 4x batch. That learning rate is the one setting this model has not trained at.
- **Validation at r = 16.** Training samples up to 16 loops. A recurrent-depth model is supposed to improve, or at least hold, with more loops at inference. The 50M model got worse past r = 4, so this column tracks whether more data changes that.
- **Every checkpoint kept.** Benchmarks can then be scored against tokens seen, every 8.2M tokens.

With fused experts, the 50M run trained at about 12,000 tokens/s, 13,300 at its end, or about 70 minutes per epoch. At that rate 500M tokens take about 10.4-11.6 h, plus about 20 minutes of validation.

### Before launching

`scripts/whitepaper.sh` ([whitepaper.md](whitepaper.md)) runs this as its `wp-500m-loop` stage. It writes `runs/whitepaper/configs/pretrain-500m-loop.toml` from this config, with the learning rate and loop depth its 50M runs chose, and it waits out CUDA failures. Item 2 below is then done by its 50M learning-rate pair, and item 3 by its loop-mean-8 run. For a run outside that pipeline:


1. **CUDA check (2 minutes).** After a suspend, CUDA fails until the driver is reloaded (`sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`) or the machine reboots. The script stops on that error rather than retrying. Twenty steps with validation at step 10 and 20 confirm the speed and the memory, including r = 16. A config must sit inside the repo, because its paths resolve from the repo root.

   ```bash
   sed -e 's|^run_dir = .*|run_dir = "runs/smoke-500m"|' -e 's|^val_every = .*|val_every = 10|' \
       configs/pretrain-500m.toml > runs/smoke-500m.toml
   ./target/release/loopmind pretrain --config runs/smoke-500m.toml --max-steps 20
   rm -rf runs/smoke-500m runs/smoke-500m.toml
   ```

   On 2026-11-28 the CPU version of this check passed: one step, validation at all five loop counts, and a 114 MB checkpoint. The GPU version waits on a driver reload.

2. **Learning-rate check (about 30 minutes, optional).** Run 300 steps at each peak rate, and keep 6e-4 if its validation loss at step 250 is lower and its gradient norm has no spikes. `--max-steps` stops early on the full run's schedule, so these steps match the real run's first 300.

   ```bash
   sed 's|runs/pretrain-500m|runs/lr-check-6e-4|' configs/pretrain-500m.toml > runs/lr-6e-4.toml
   sed -e 's|runs/pretrain-500m|runs/lr-check-3e-4|' -e 's|^learning_rate = 6.0e-4|learning_rate = 3.0e-4|' \
       -e 's|^min_learning_rate = 6.0e-5|min_learning_rate = 3.0e-5|' configs/pretrain-500m.toml > runs/lr-3e-4.toml
   ./target/release/loopmind pretrain --config runs/lr-6e-4.toml --max-steps 300
   ./target/release/loopmind pretrain --config runs/lr-3e-4.toml --max-steps 300
   ```

3. **Loop depth.** If a 50M-token comparison shows that a deeper loop mean makes extra loops help, set `loop_count_mean` in the model config before this run. A mean of 8 runs more forward passes but the same 4 backward passes, about a quarter slower.

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
