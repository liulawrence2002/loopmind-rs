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

### The 500M mix

`configs/data-500m.toml` builds the data for the 500M run ([pretrain.md](pretrain.md#the-500m-run)) into `data/prepared-500m`. It lists its sources in `[[pretrain.mix]]` tables, each with a whole-percent share, in build order:

```toml
[[pretrain.mix]]
source = "fineweb"
percent = 53
```

Each source gets `total_tokens * percent / 100`, and the last one absorbs the remainder. The shares must sum to 100. Each source may appear once and must be one of `fineweb`, `cosmopedia`, `tinystories`, `fable-wiki`, or `simplewiki`. Without `[[pretrain.mix]]`, the build uses the 50/20/20/10 mix above. The field is then left out of the fingerprint, so `configs/data.toml` builds are unaffected.

| Source | Dataset | Share | Budget | Licence |
| --- | --- | ---: | ---: | --- |
| fineweb | `HuggingFaceFW/fineweb-edu`, `sample-10BT` | 53% | 265M | ODC-By |
| cosmopedia | `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2` | 25% | 125M | ODC-By |
| simplewiki | `wikimedia/wikipedia`, `20231101.simple`, revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | 10% | 50M | CC-BY-SA-3.0 and GFDL |
| tinystories | `roneneldan/TinyStories` | 10% | 50M | CDLA-Sharing-1.0 |
| fable-wiki | `KeremTT/FABLE-5.2-MAX-DATABASE`, `raw_text_corpus`, English | 2% | 10M | per row (`other` on the card) |

The mix moves from 50/20/20/10 for three reasons:

- **FABLE's text corpus is small.** It has 10,000 English rows, about 16M tokens with this tokenizer. That is too few for 10% of 500M, so it drops to 2%.
- **Simple English Wikipedia adds facts.** Its 241,787 articles define things in plain language, such as "April is the fourth month of the year". The articles are short (median 514 characters) and total about 75M tokens, so 50M leaves room.
- **Fewer stories.** TinyStories drops from 20% to 10%, which is still five times the 50M run's stories. The target is answering questions, not telling stories, so Cosmopedia's textbook-style text rises to 25%.

Every source streams from its first row. A smaller `total_tokens` with the same mix therefore reads a prefix of each source, so a 50M build of this mix holds the same documents as the start of the 500M one.

Measured on the 4070 box on 2026-11-28, the build took 631.6 s and wrote 968 MB:

| Source | Train tokens | Val tokens | Documents read |
| --- | ---: | ---: | ---: |
| fineweb | 264,999,936 | 2,649,088 | 206,035 |
| cosmopedia | 124,999,680 | 1,249,280 | 134,931 |
| simplewiki | 49,999,872 | 499,712 | 137,962 |
| tinystories | 49,999,872 | 499,712 | 208,896 |
| fable-wiki | 9,999,360 | 99,328 | 6,279 |
| **Total** | **499,998,720** | **4,997,120** | **694,103** |

Simple English Wikipedia used 137,962 of its 241,787 articles, and FABLE 6,279 of its 10,000 English rows.

## SFT

English chat rows from FABLE, excluding `raw_text_corpus` and the `turkish_language` config, then optionally MiniMind's and smol-smoltalk's English rows, in that order. A FABLE row's `language` must be `en` or `english`. Rows are rendered with the M2 chat template (`<|bos|>`, role turns, `<|eos|>` when the transcript ends on an assistant turn). The loss mask is 1 on assistant content, the `<|im_end|>` that closes it, and that final `<|eos|>`.

### English only

Every source goes through one English check, `language_of` in `python/stream_corpus.py`, before the Rust side keeps a row. The check judges the answers. A row is dropped as `non_english` when:

- an assistant answer is more than 30% CJK characters;
- an assistant answer is more than 30% other non-Latin script (Greek, Cyrillic, Arabic, Hebrew, Indic, Thai, Georgian);
- an assistant answer of 15 or more words is mostly Spanish, French, German, Italian, Portuguese, or Dutch: function words from those languages are over 15% of its words, there are at least 4 different ones, and they outnumber English function words;
- the whole conversation is more than 2% CJK.

These pass: a name like 秦始皇, a π, or a Russian name inside an English answer; a list of Dutch names or Spanish restaurant names; and JSON. The 2% conversation line drops Chinese chats with short answers, and MiniMind's English rows whose tool definitions are in Chinese. Those rows sit at 2–10% CJK, while incidental characters stay under 2%.

Applied to FABLE's English-tagged rows, the check drops 162 rows and none of the eval rows. They are French dialogue, code with Chinese comments, SQL on Chinese strings, and tool chats that answer in another language. FABLE rows must also be tagged `en`.

Two stricter first versions were too broad. One checked prompts as well, at 0% CJK. The other used thresholds of 5% CJK and 10% other script in answers, with foreign function words at 5% of the words. They dropped 1,064 and 611 FABLE rows with English answers, mostly JSON, name lists, and answers that quote a name in its own script. They also changed the eval set.

### Eval overlap

While FABLE streams, the user prompts of its four checkable configs are collected. That is about 60,000 prompts, a superset of the eval rows. A MiniMind or smol-smoltalk row whose user prompt matches one is dropped, counted as `eval_overlap`. Prompts are compared after lowercasing and collapsing whitespace.

### Long conversations

FABLE rows over `sft.max_tokens` are dropped, as before. MiniMind and smol-smoltalk rows are trimmed instead: whole trailing turns are removed until the conversation fits and ends on an assistant answer. Rows trimmed this way are counted as `trimmed`. A row whose first answer alone does not fit is still `too_long`.

### MiniMind

When `sft.minimind_file` is set, the English rows of [MiniMind](https://github.com/jingyaogong/minimind)'s SFT mix follow FABLE's in the same build, as config `minimind_sft`.

| Setting | Value |
| --- | --- |
| Dataset | `jingyaogong/minimind_dataset` |
| Revision | `revisions.minimind`, `312afb4f76391145c6902f765bb51691c09a12f5` |
| File | `sft_t2t_mini.jsonl`, MiniMind's quick-run mix: 905,718 rows, 1.74 GB, SHA-256 `abb1e76b…fa518` |
| Full mix | `sft_t2t.jsonl`, 14.1 GB; set `sft.minimind_file` to it to use it |
| Local path | `data/raw/minimind/`; the stream downloads the file there when it is missing |

Each row is `{"conversations": [{"role", "content", ...}]}`. `python/stream_corpus.py` turns it into FABLE's form:

- **Language:** MiniMind has no language field. The English check above runs on the rendered text, tool definitions included. Most rows are Chinese, and Chinese costs this byte-level BPE three tokens per character.
- **Tools:** `tools` (JSON strings) go into the system turn as FABLE does it: FABLE's tool instruction, then `<tools>[...]</tools>`. Each `tool_calls` entry becomes `<tool_call>{"name", "arguments"}</tool_call>` in the assistant turn, and `tool` turns are wrapped in `<tool_response></tool_response>`.
- **Reasoning:** `reasoning_content` is dropped, so the model learns the final answers.
- **Empty answers:** a row with an empty assistant answer and no tool call is sent empty, and is counted as `empty`.

The eval split is drawn from FABLE's four checkable configs only, and the other sources follow FABLE's rows in the stream.

The dataset card lists two licences, Apache-2.0 and CC-BY-NC-2.0, without saying which rows carry which. Treat a build that includes MiniMind as non-commercial. Its rows are recorded with the license string `apache-2.0 / cc-by-nc-2.0 (minimind_dataset)`.

### smol-smoltalk

With `sft.smoltalk = true`, the train split of [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) follows MiniMind's rows. It is the SFT mix Hugging Face used for SmolLM2-135M-Instruct and SmolLM2-360M-Instruct (Allal et al., 2025). It is a version of SmolTalk adapted for models under 1B parameters: shorter Magpie-Ultra conversations, no function calling, and no advanced math.

| Setting | Value |
| --- | --- |
| Dataset | `HuggingFaceTB/smol-smoltalk`, Apache-2.0, `language: en` |
| Revision | `revisions.smoltalk`, `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc` |
| Files | `data/train-0000{0..3}-of-00004.parquet`, 460,341 rows. The stream fetches them once into the Hugging Face cache with `snapshot_download`. |
| Config name | `smoltalk:<source>`, so the index keeps each row's source |

| Source | Rows | Share |
| --- | ---: | ---: |
| smol-magpie-ultra-short | 270,838 | 58.8% |
| self-oss-instruct | 48,071 | 10.4% |
| openhermes-50k | 47,492 | 10.3% |
| smol-contraints | 34,433 | 7.5% |
| smollm-rewrite-30k | 26,657 | 5.8% |
| smol-summarize-20k | 19,272 | 4.2% |
| smol-summarize-5k | 4,749 | 1.0% |
| longalign | 3,560 | 0.8% |
| explore-instruct-rewrite | 3,017 | 0.7% |
| everyday-conversations | 2,252 | 0.5% |

Before the build, 1,267 rows (0.3%) failed the English check, mostly in openhermes-50k: translations and tasks in other languages. 27 rows repeated an eval prompt. 59% of rows are three-turn Magpie chats. Only 47% of rows are under about 3,300 characters, which is roughly 1,024 tokens, so trimming keeps the rest as their opening turns.

Kept conversations are at most `sft.max_tokens` (1024). Drop reasons recorded in the manifest: `non_english`, `cc_by_sa`, `too_long`, `empty`, `no_assistant`, `bad_role`, `eval_overlap`; `trimmed` counts kept rows that lost trailing turns. CC-BY-SA rows (license string containing `cc-by-sa`, case insensitive) are kept unless `sft.exclude_cc_by_sa` is true. `english_license_counts` counts every English row, including ones dropped later for length or license.

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

### With MiniMind

Measured on the 4070 box, 2026-11-27, with `sft.minimind_file = "data/raw/minimind/sft_t2t_mini.jsonl"`. The build took 149.8 s.

| Outcome | FABLE only | FABLE + MiniMind |
| --- | ---: | ---: |
| seen | 200,000 | 1,105,718 |
| kept train | 153,517 | 255,643 |
| kept eval | 2,000 | 2,000 |
| dropped non_english | 27,413 | 830,596 |
| dropped too_long | 17,070 | 17,446 |
| dropped empty | 0 | 33 |

| MiniMind rows | Count |
| --- | ---: |
| in the file | 905,718 |
| with CJK text, dropped | 803,183 |
| English | 102,502 |
| English, over 1,024 tokens | 376 |
| English, kept for train | 102,126 |

31,753 of the CJK drops have English messages but Chinese tool definitions or calls.

| Train split | Rows | Tokens | Mean tokens | Padded to 64 |
| --- | ---: | ---: | ---: | ---: |
| FABLE | 153,517 | 51.0M | 332 | 56.0M |
| MiniMind | 102,126 | 22.8M | 224 | 26.1M |
| total | 255,643 | 73.8M | 289 | 82.0M |

- The English MiniMind rows are general instruction and chat. 98% are single-turn (100,586 of 102,502). One has a tool call.
- `eval.jsonl` has the same SHA-256 as the FABLE-only build (`1c17344a…`).
- An SFT epoch is 15,978 steps of 16 rows, up from 9,595.

### With MiniMind and smol-smoltalk

This is the build `configs/data.toml` makes now, with `sft.minimind_file` and `sft.smoltalk = true`, measured on the 4070 box on 2026-11-27. It took 960.7 s. Most of that is the Rust side re-rendering long chats while it trims them.

| Outcome | Rows |
| --- | ---: |
| seen | 1,566,059 |
| kept train | 699,885 |
| kept eval | 2,000 |
| dropped non_english | 829,868 |
| dropped too_long | 28,473 |
| dropped eval_overlap | 5,800 |
| dropped empty | 33 |
| trimmed to fit, kept | 220,971 |

| Train split | Rows | Tokens | Mean tokens | Padded to 64 | Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| FABLE | 153,396 | 51.0M | 332 | 55.9M | 15.1% |
| MiniMind | 99,265 | 22.1M | 222 | 25.2M | 6.8% |
| smol-smoltalk | 447,224 | 276.0M | 617 | 289.9M | 78.1% |
| total | 699,885 | 349.1M | 499 | 371.0M | |

smol-smoltalk rows kept, by source:

| Source | Rows kept |
| --- | ---: |
| smol-magpie-ultra-short | 270,563 |
| self-oss-instruct | 47,648 |
| openhermes-50k | 43,059 |
| smol-contraints | 34,222 |
| smollm-rewrite-30k | 26,646 |
| smol-summarize-20k | 15,114 |
| smol-summarize-5k | 4,703 |
| explore-instruct-rewrite | 3,017 |
| everyday-conversations | 2,252 |
| longalign | 0 |

All `longalign` rows are long-document tasks whose first answer alone is over 1,024 tokens. `eval.jsonl` is byte-identical to the FABLE-only build. An SFT epoch is 43,743 steps of 16 rows.

`prepare-sft` checks the build fingerprint, which covers `configs/data.toml` and the tokenizer but not `python/stream_corpus.py`. After changing the stream script, delete `data/prepared/sft` before rebuilding.

## Running it on the ThinkPad

Prepare is a CPU tokenizer job. The RTX 4070 Laptop is not used, and CUDA is not required.

Lawrence's Windows ThinkPad can run the same commands. WSL2 (Ubuntu) is the path that matches this repo's Linux CI: install Rustup, `g++` (the `tokenizers` crate builds the esaxx C++ code), Python 3.12, and `pip install datasets huggingface_hub`. From the repo root:

```bash
cargo run -p loopmind --release -- prepare-pretrain --config configs/data.toml
cargo run -p loopmind --release -- prepare-sft --config configs/data.toml
```

Run the same command again to resume. Expect the Hugging Face cache plus a few hundred MB under `data/prepared/` for the 50M mix, and on the order of a gigabyte once SFT staging is included. `configs/data.toml` sets `python = "python3"`. On a native Windows prompt without WSL, install the Visual Studio C++ build tools as well, and set `python` to whatever launches Python 3 (`python` or `py`). The value is split on whitespace, so `py -3` works.

Raise `pretrain.total_tokens` to `200000000` for a 200M mix. Change `output_dir` when you do, or delete `data/prepared/`, because a new budget does not match the old fingerprint.
