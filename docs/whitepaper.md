# LoopMind: a looped mixture-of-experts core at 10M parameters

*Draft.* The method and setup below describe the code as it is. Sections marked **[pending]** fill in from `runs/whitepaper/results.md`, which `scripts/whitepaper.sh` regenerates after every stage. The decision rules in [Section 4.5](#45-decision-rules) were fixed before any of these runs started.

## Abstract

LoopMind is a 9.9M-parameter language model that runs one shared two-layer core, a sparse mixture of experts, a variable number of times per token. It follows the recurrent-depth design of Huginn (Geiping et al., 2025), with MoE feed-forward layers in the core as in MoEUT (Csordás et al., 2024). Each token activates 2.4M parameters per pass, and the loop count `r` is drawn at random during training and chosen freely at inference.

We ask three questions at this scale:
1. At equal parameters, does looping the core beat running it once?
2. At equal compute, how does a looped shared core compare with a standard transformer whose layers are all distinct?
3. Does accuracy keep improving when the model loops more at inference than it usually did in training?

The comparison runs train on 50M tokens; the main runs train on 500M. Evaluation is zero-shot log-likelihood scoring on eight standard benchmarks. **[pending: results.]**

## 1. Introduction

Small language models are limited by parameters. Two lines of work loosen the tie between parameters and compute in opposite directions:
- **Mixtures of experts** store many feed-forward blocks but run few per token (Shazeer et al., 2017; Fedus et al., 2022).
- **Weight sharing across depth** runs the same layers several times. Examples are the Universal Transformer (Dehghani et al., 2019), ALBERT (Lan et al., 2020), and, with a variable loop count trained for test-time compute, Huginn (Geiping et al., 2025).

MoEUT combines the two, so that shared layers stay wide without costing more per token (Csordás et al., 2024).

These results come from models of tens of millions to billions of parameters. LoopMind tests the combination at about 10M parameters, the scale of TinyStories models (Eldan and Li, 2023) and the smallest Pythia models (Biderman et al., 2023). At that size every parameter is scarce. If recurrence pays for itself anywhere, it should be here.

The whole pipeline is written in Rust on Burn and trains on a single 8 GB laptop GPU: tokenizer, data, pretraining, SFT, evaluation, and a local chat server.

## 2. Architecture

| Stage | Layers | Feed-forward |
| --- | --- | --- |
| Prelude | 1 | dense SwiGLU, width 768 |
| Core | 2, shared across loop passes | 32 SwiGLU experts of width 128, top-2, routed again every pass |
| Coda | 1 | dense SwiGLU, width 768 |

Each layer is pre-norm: RMSNorm, attention, residual, RMSNorm, feed-forward, residual. Attention is grouped-query, with `d_model` 256, 4 query heads, 2 key/value heads, head dimension 64, RoPE with θ = 10⁶, and no biases. The vocabulary is 6,400 byte-level BPE tokens, trained on English, and the context is 1,024 tokens. Embeddings are tied with the output head.

The prelude maps the embedded tokens to a state `e`. The core then runs `r` times, starting from `s = e` and computing `s ← Core(s) + e` on each pass, so every pass sees the input again. The coda and a final RMSNorm map `s` to logits. The router picks each token's top two experts again on every pass, so a token can use different experts at different depths. A Switch-style load-balancing loss (Fedus et al., 2022), weighted 0.01, keeps the experts in use.

**Training the loop.** Each training sequence gets its own `r`, drawn from a Poisson distribution with mean 4 and clamped to 1–16. Gradients flow through the last 4 passes only. Earlier passes run without an autodiff graph, and their result enters as a constant. Memory therefore does not grow with `r`. Because each sequence has its own `r`, the micro-batch is one sequence.

**Parameters.** The model has 9,914,624 parameters. Of these, 1,638,400 are the tied embedding and 6,291,456 are experts, of which 393,216 are active per token. That leaves 2,377,984 active parameters per token per pass, excluding the embedding. Running the core `r` times multiplies compute, not parameters: the model does `2 + 2r` layer passes per token, or 10 at `r = 4`.

## 3. Comparison models

| Model | Params | Active | Layer passes per token | Training `r` |
| --- | ---: | ---: | ---: | --- |
| LoopMind (mean 4) | 9.91M | 2.38M | 2 + 2r (10 at r = 4) | Poisson(4), 1–16 |
| LoopMind (mean 8) | 9.91M | 2.38M | 2 + 2r | Poisson(8), 1–32 |
| No recurrence | 9.91M | 2.38M | 4 | 1 |
| Unshared, 10 layers | 30.0M | 4.79M | 10 | 1 |

- **No recurrence** has LoopMind's exact layers and is trained with `r = 1`. It isolates what looping adds at fixed parameters.
- **Unshared** gives each of the 10 layer passes its own weights: 8 distinct core layers, run once. It matches LoopMind's compute at `r = 4` with three times the parameters. It is the usual way to spend the same compute.
- **LoopMind (mean 8)** trains with deeper loops. It tests whether inference beyond `r = 4` needs deeper training. In the first 50M-token run, held-out loss was lowest at `r = 4` and higher at `r = 8`. After SFT it was much higher at `r = 16`.

All four use the same tokenizer, data, optimizer, and schedule. They differ only in the model config: [configs/whitepaper/](../configs/whitepaper/).

## 4. Setup

### 4.1 Pretraining data

The mix is FineWeb-Edu 53%, Cosmopedia v2 25%, Simple English Wikipedia 10%, TinyStories 10%, and FABLE's English text corpus 2%. FineWeb-Edu is from Penedo et al. (2024), Cosmopedia from Ben Allal et al. (2024), and TinyStories from Eldan and Li (2023). Every dataset revision is pinned. The comparison runs use the first 50M tokens of each source in proportion; the main runs use 500M. Each source streams from its first row, so the 50M build is a prefix of the 500M one. Details are in [data.md](data.md#the-500m-mix).

### 4.2 Optimization

| Setting | Value |
| --- | --- |
| Optimizer | AdamW, β = (0.9, 0.95), ε = 10⁻⁸, weight decay 0.1 |
| Batch | 32 sequences of 1,024 tokens (32,768 tokens) |
| Learning rate | peak 6e-4 or 3e-4 ([Section 4.5](#45-decision-rules)), linear warmup, cosine to a tenth of peak |
| Warmup | 100 steps at 50M tokens, 300 at 500M |
| Steps | 1,526 at 50M tokens, 15,259 at 500M |
| Gradient clipping | global norm 1.0 |
| Precision | f32 weights and optimizer state |
| Hardware | one RTX 4070 Laptop GPU (8 GB); Rust 1.100 with Burn 0.22 on CUDA |

Each run uses seed 42 and one epoch. The main 50M pair, LoopMind (mean 4) and no recurrence, is also trained at seed 7, which changes the initialization, the data order, and the sampled loop counts. Validation measures next-token loss on 64 held-out pretrain sequences at fixed `r`.

### 4.3 Benchmarks

`loopmind bench` scores eight zero-shot tasks by log-likelihood, with the prompt formats of lm-evaluation-harness (Gao et al.). Each answer choice is scored by the summed log-probability of its tokens after the context.
- **`acc`** picks the choice with the highest sum.
- **`acc_norm`** divides each sum by the choice's length in bytes.
- **LAMBADA** has a single target word, scored as right when greedy decoding reproduces it.

| Task | Items | Chance | Reported |
| --- | ---: | ---: | --- |
| ARC-Easy (Clark et al., 2018) | 2,376 | 25% | acc_norm |
| ARC-Challenge | 1,172 | 25% | acc_norm |
| PIQA (Bisk et al., 2020) | 1,838 | 50% | acc_norm |
| HellaSwag (Zellers et al., 2019) | 10,042 | 25% | acc_norm |
| SciQ (Welbl et al., 2017) | 1,000 | 25% | acc_norm |
| OpenBookQA (Mihaylov et al., 2018) | 500 | 25% | acc_norm |
| LAMBADA (Paperno et al., 2016) | 5,153 | 0% | acc |
| BLiMP (Warstadt et al., 2020) | 67,000 | 50% | acc |

The headline number is the mean over the eight tasks. At 10M parameters, several tasks sit near chance, so per-task results matter more than the mean. Every looped model is scored at `r` = 1, 2, 4, 8, 16, and 32. Dataset revisions are pinned in `python/fetch_benchmarks.py`.

### 4.4 Instruction tuning

Both 500M models get the same SFT: 8,000 steps of 16 conversations, with a cosine learning rate from 1e-4 to 1e-5. That is 128,000 conversations, 21% of the 600,620-row training split, a seeded random sample of FABLE's English chat and smol-smoltalk (Ben Allal et al., 2025), filtered to English answers, with any row that repeats an eval prompt dropped. MiniMind's data is left out because its licence includes CC-BY-NC. The tuned models are scored on the same benchmarks, plus FABLE's 2,000 held-out questions in math, logic, science, and JSON ([sft.md](sft.md), [refine.md](refine.md)).

### 4.5 Decision rules

`scripts/whitepaper.sh` makes two choices and records them in `runs/whitepaper/choices.env`:
1. **Learning rate.** LoopMind (mean 4) trains on 50M tokens at peak 6e-4 and at 3e-4. The rate with the lower final validation loss at `r = 4` is used for every later run.
2. **Loop depth for 500M.** The mean-8 model replaces mean 4 only if its best benchmark mean over `r` beats mean 4's by at least 0.5 points. Otherwise mean 4 stays, since it trains faster.

## 5. Results

### 5.1 Looping at equal parameters

**[pending]** LoopMind (mean 4) against no recurrence at 50M tokens: validation loss and benchmarks. This is "50M tokens: benchmarks" in `results.md`.

### 5.2 Shared and unshared depth at equal compute

**[pending]** LoopMind at `r = 4` against the unshared 10-layer model, with three times the parameters.

### 5.3 More loops at inference

**[pending]** Benchmark mean and validation loss against `r` for mean 4 and mean 8 ("Test-time loops" in `results.md`). The question is whether `r` above the training mean helps, holds, or hurts, and whether deeper training changes that.

### 5.4 500M tokens

**[pending]** LoopMind against no recurrence at 500M tokens: final benchmarks, validation loss by `r`, and learning curves every 49M tokens.

### 5.5 Seed spread

**[pending]** The main 50M pair at seeds 42 and 7: how far a rerun moves each model's benchmark mean.

### 5.6 After instruction tuning

**[pending]** Benchmarks, FABLE held-out scores, and chat samples for both SFT models.

## 6. Limitations

- **Few seeds.** Only the main 50M pair has a second seed, and that gives the run-to-run spread. Differences smaller than it, often a point or two on these benchmarks, are noise.
- **Tokens are fixed, not compute.** Comparisons hold the token count fixed. LoopMind at `r = 4` spends about 2.5 times the training compute per token of the no-recurrence model, which is why the unshared model, matched on compute, is included.
- **Near-chance tasks.** At 10M parameters, ARC-Challenge, OpenBookQA, and HellaSwag sit near chance for every model, so they say little about the architecture.
- **Short SFT.** The instruction tuning is 8,000 steps over a sample of two English datasets. FABLE's held-out questions favor its own answer formats.
- **Laptop hardware.** Throughput figures come from one laptop GPU in f32, with batch-1 micro-steps, because each sequence has its own `r`. They do not reflect what the architecture could reach on server hardware.

## 7. Reproducing

```bash
cargo build --release -p loopmind --features cuda
./target/release/loopmind prepare-pretrain --config configs/whitepaper/data-mix50m.toml
./target/release/loopmind prepare-pretrain --config configs/data-500m.toml
./target/release/loopmind prepare-sft --config configs/whitepaper/data-sft.toml
python python/fetch_benchmarks.py --out data/bench
setsid nohup scripts/whitepaper.sh > /dev/null 2>&1 &
cat runs/whitepaper/STATUS
```

The pipeline resumes where it stopped when run again. After a laptop suspend, CUDA fails until the NVIDIA driver is reloaded (`sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`) or the machine reboots. While CUDA is down, the pipeline waits, checking every 5 minutes.

Every run's metrics are in `runs/<run>/metrics.jsonl`. Benchmarks are in `runs/bench/<stage>.json`, and the results tables in `runs/whitepaper/results.md`.

## References

- Ben Allal, L., Lozhkov, A., Penedo, G., Wolf, T., and von Werra, L. (2024). Cosmopedia. Hugging Face.
- Ben Allal, L., et al. (2025). SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model. arXiv:2502.02737.
- Biderman, S., et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. ICML.
- Bisk, Y., et al. (2020). PIQA: Reasoning about Physical Commonsense in Natural Language. AAAI.
- Clark, P., et al. (2018). Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457.
- Csordás, R., Irie, K., Schmidhuber, J., Potts, C., and Manning, C. D. (2024). MoEUT: Mixture-of-Experts Universal Transformers. NeurIPS.
- Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., and Kaiser, Ł. (2019). Universal Transformers. ICLR.
- Eldan, R., and Li, Y. (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent English? arXiv:2305.07759.
- Fedus, W., Zoph, B., and Shazeer, N. (2022). Switch Transformers. JMLR.
- Gao, L., et al. A framework for few-shot language model evaluation (lm-evaluation-harness). EleutherAI.
- Geiping, J., et al. (2025). Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. arXiv:2502.05171.
- Lan, Z., et al. (2020). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. ICLR.
- Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. (2018). Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering. EMNLP.
- Paperno, D., et al. (2016). The LAMBADA dataset: Word prediction requiring a broad discourse context. ACL.
- Penedo, G., et al. (2024). The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. NeurIPS Datasets and Benchmarks.
- Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR.
- Warstadt, A., et al. (2020). BLiMP: The Benchmark of Linguistic Minimal Pairs for English. TACL.
- Welbl, J., Liu, N. F., and Gardner, M. (2017). Crowdsourcing Multiple Choice Science Questions. W-NUT.
- Zellers, R., et al. (2019). HellaSwag: Can a Machine Really Finish Your Sentence? ACL.
