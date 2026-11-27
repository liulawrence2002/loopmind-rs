# Data pipeline

`configs/data.toml` is the contract for the pretrain mix and the English SFT split. The commands are:

```bash
cargo run -p loopmind --release -- prepare-pretrain --config configs/data.toml
cargo run -p loopmind --release -- prepare-sft --config configs/data.toml
```

Both commands are idempotent. A finished manifest whose fingerprint matches the config and `tokenizer/tokenizer.json` is left untouched. An interrupted pretrain source resumes from the last flushed shard: the Python stream is skipped forward by the saved document count, and the partial sequence is kept. An interrupted SFT scan resumes from the last staged checkpoint. A fingerprint mismatch refuses to append; point `output_dir` somewhere else.

Raw downloads stay under the Hugging Face cache and under `data/`, which is gitignored. The committed artifacts are the config, the streamer, and this note. CI builds fixture shards in memory and does not download these datasets.

## Pretrain

The mix is 50% FineWeb-Edu, 20% Cosmopedia-v2, 20% TinyStories, and 10% English `raw_text_corpus` rows from FABLE. `pretrain.total_tokens` scales that ratio. `200_000_000` is 100M / 40M / 40M / 20M.

| Source | Dataset | Config | Revision |
| --- | --- | --- | --- |
| fineweb | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| cosmopedia | `HuggingFaceTB/smollm-corpus` | `cosmopedia-v2` | `3ba9d605774198c5868892d7a8deda78031a781f` |
| tinystories | `roneneldan/TinyStories` | train split | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |
| fable-wiki | `KeremTT/FABLE-5.2-MAX-DATABASE` | `raw_text_corpus`, `language=en` | `d0bcd6b2181f5588afa9e136a383397dff956620` |

Each document is tokenized with `tokenizer/tokenizer.json` and followed by `<|eos|>`. Sequences are exactly `context_length` tokens (1024). A source stops at the largest multiple of 1024 that does not exceed its budget, so the 50M setting packs **49,997,824** train tokens (2,176 under the headline budget):

| Source | Budget | Packed train tokens |
| --- | ---: | ---: |
| fineweb | 25,000,000 | 24,999,936 |
| cosmopedia | 10,000,000 | 9,999,360 |
| tinystories | 10,000,000 | 9,999,360 |
| fable-wiki | 5,000,000 | 4,999,168 |

Validation is extra, not taken out of the train budget. A stable hash of `(seed, source, document index)` sends about `val_ratio` (default 1%) of documents to val. Val is also floored to a multiple of 1024. Seed is 42.

Shards are little-endian `u16` files (`vocab` is 6400). The header is magic `LMDP`, version `u16`, sequence length `u16`, sequence count `u32`, then `count * seq_len` token ids. `pretrain/manifest.json` records the token counts, shard paths, and sha256 of each file. Default shard size is 512 sequences.

## SFT

English chat rows from FABLE, excluding `raw_text_corpus` and the `turkish_language` config. `language` must be `en` or `english`. Rows are rendered with the M2 chat template (`<|bos|>`, role turns, `<|eos|>` when the transcript ends on an assistant turn). The loss mask is 1 on assistant content, the `<|im_end|>` that closes it, and that final `<|eos|>`.

Kept conversations are at most `sft.max_tokens` (1024). Drop reasons recorded in the manifest: `non_english`, `cc_by_sa`, `too_long`, `empty`, `no_assistant`, `bad_role`. CC-BY-SA rows (license string containing `cc-by-sa`, case insensitive) are kept unless `sft.exclude_cc_by_sa` is true. `english_license_counts` counts every English row, including ones dropped later for length or license.

Eval holds out `sft.eval_rows` (2000) checkable rows: `math_reasoning`, `logic_reasoning`, `science_knowledge`, and `structured_output`, 500 from each config, the lowest stable hashes first. If one config is short, the remainder is filled from the other three. Those rows are not in the train shards. `sft/eval.jsonl` stores `id`, `config`, `license`, `reference` (last assistant text), `tokens`, and `loss_mask`.

Train shards use magic `LMDS`: version, row count, then for each row a `u16` length, the token ids, and the mask bytes.

## Dataloader

`loopmind-data` samples shards into plain `u16` batches. `loopmind-model::pretrain_batch` and `sft_batch` copy a batch into Burn tensors, so the data crate does not depend on Burn.

Pretrain sampling picks a random packed sequence (the loader seeks to that sequence; it does not read the whole shard). SFT sampling picks a length bucket (64, 128, 256, 512, or 1024), draws rows from that bucket, and pads with `<|pad|>`. Pad positions have mask 0. Both samplers take a seed.

## Measured build

Release CLI on this VM, `configs/data.toml` unchanged, datasets streamed at the pinned revisions. A second run of each command returned in 0.0s without rewriting shards.

Pretrain wall time was **44.6s**. Train tokens are the packed budgets above (48,826 sequences). Validation is the extra 1% floor.

| Source | Docs read | Train tokens | Train seqs | Val tokens | Val seqs |
| --- | ---: | ---: | ---: | ---: | ---: |
| fineweb | 19,488 | 24,999,936 | 24,414 | 249,856 | 244 |
| cosmopedia | 10,806 | 9,999,360 | 9,765 | 99,328 | 97 |
| tinystories | 42,167 | 9,999,360 | 9,765 | 99,328 | 97 |
| fable-wiki | 3,151 | 4,999,168 | 4,882 | 49,152 | 48 |
| total | 75,612 | 49,997,824 | 48,826 | 497,664 | 486 |

On disk: train shards 99,996,824 bytes (98 files), val shards 995,376 bytes (4 files), state plus manifest about 43 KB. `data/prepared/pretrain` is 101,035,087 bytes. A full shard is 512 sequences: magic, version, length, count, then `512 * 1024` little-endian `u16` ids (1,048,588 bytes).

SFT wall time was **82.3s**. The ten chat configs are 200,000 rows. English rows by license (counted before the length drop):

| License | English rows |
| --- | ---: |
| MIT | 48,426 |
| Apache-2.0 | 37,827 |
| CC-BY-4.0 | 23,618 |
| CC0-1.0 | 21,346 |
| CC-BY-SA-4.0 | 20,370 |
| CC-BY-SA-3.0 | 15,000 |
| ODC-BY-1.0 | 6,000 |

| Outcome | Rows |
| --- | ---: |
| seen | 200,000 |
| kept train | 153,517 |
| kept eval | 2,000 |
| dropped non_english | 27,413 |
| dropped too_long | 17,070 |
| dropped cc_by_sa | 0 |
| dropped empty | 0 |
| dropped no_assistant | 0 |
| dropped bad_role | 0 |

Eval is 500 rows from each of `math_reasoning`, `logic_reasoning`, `science_knowledge`, and `structured_output`. Every eval row has a non-empty reference. CC-BY-SA rows were kept. The 27,413 non-English drops are rows inside those ten configs; `turkish_language` and `raw_text_corpus` are not read.

On disk: train shards 153,426,899 bytes (600 files, 256 rows each, last shard shorter), `eval.jsonl` 3,768,063 bytes, staging checkpoint 237,751,964 bytes, manifest 105,652 bytes. `data/prepared/sft` is 395,052,578 bytes. Staging can be deleted after the manifest exists; a later run trusts the manifest. `data/prepared` together is 496,087,665 bytes.

Train shards use magic `LMDS`. SFT sampling pads to 64, 128, 256, 512, or 1024. That sampler is what CI exercises, on synthetic shards, not on this download.

## Running it on the ThinkPad

Prepare is a CPU tokenizer job. The RTX 4070 Laptop is not used, and CUDA is not required.

Lawrence's Windows ThinkPad can run the same commands. WSL2 (Ubuntu) is the path that matches this repo's Linux CI: install Rustup, `g++` (the `tokenizers` crate builds the esaxx C++ code), Python 3.12, and `pip install datasets huggingface_hub`. From the repo root:

```bash
cargo run -p loopmind --release -- prepare-pretrain --config configs/data.toml
cargo run -p loopmind --release -- prepare-sft --config configs/data.toml
```

Run the same command again to resume. Expect the Hugging Face cache plus a few hundred MB under `data/prepared/` for the 50M mix, and on the order of a gigabyte once SFT staging is included. `configs/data.toml` sets `python = "python3"`. On a native Windows prompt without WSL, install the Visual Studio C++ build tools as well, and set `python` to whatever launches Python 3 (`python` or `py`). The value is split on whitespace, so `py -3` works.

Raise `pretrain.total_tokens` to `200000000` for a 200M mix. Change `output_dir` when you do, or delete `data/prepared/`, because a new budget does not match the old fingerprint.
