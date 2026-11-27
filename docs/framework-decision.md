# Burn vs Candle

M1 keeps Burn as the training framework. A Candle port of the whole model is not worth it on the evidence below.

## What was timed

One core MoE layer, the default shape: `d_model` 256, 32 SwiGLU experts of width 128, top-2 routing, 128 tokens. Each step is a forward plus a backward. Both sides run all 32 experts. Weights are initialized separately. The router penalty on the Candle side is the sum of squared mean probabilities; LoopMind's own layer uses the Switch balance loss. That reduction is small next to the expert matmuls.

Command, release build, median of 5 iterations after 2 warmup iterations:

```bash
cargo run -p moe-bakeoff --release
```

| Backend | Device | Median step |
| --- | --- | ---: |
| Burn 0.22.1, `flex` | CPU | 29.80 ms |
| Candle 0.9.2 | CPU | 38.60 ms |

Burn was about 1.3× faster on this CPU step.

## What this does not say

There is no CUDA number here. This environment has no GPU. The full model, the optimizer, and a real training step were not timed. A later profile on the RTX 4070 Laptop could change the CUDA ranking; it would not by itself justify rewriting the modules that already train.

## Recommendation

Stay on Burn for M2 through M5.

- The forward pass, tied head, loop, truncation, and load-balancing loss already run on Burn.
- On the measured CPU layer, Burn is the faster of the two.
- Candle remains available for a narrower check. Reopen the question only if a CUDA profile of a full training step shows Candle ahead by enough to pay for a port.

## Run it again

CPU:

```bash
cargo run -p moe-bakeoff --release
```

CUDA, on a machine with the toolkit and an NVIDIA GPU (the dev box is an RTX 4070 Laptop, 8 GB):

```bash
cargo run -p moe-bakeoff --release --features cuda
```

That feature turns on Burn's `cuda` backend and Candle's `cuda` backend, and both devices are `cuda:0`.
