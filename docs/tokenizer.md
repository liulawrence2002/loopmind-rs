# Tokenizer

LoopMind uses a byte-level BPE with exactly 6400 tokens, the same width as the model embedding. The file is `tokenizer/tokenizer.json`. It was trained with the Hugging Face `tokenizers` crate (0.23.2) and has no unicode normalizer, so encode then decode is a byte-for-byte round trip.

## Special tokens

| Token | Role |
| --- | --- |
| `<\|bos\|>` | Start of a rendered chat |
| `<\|eos\|>` | End of a finished assistant transcript |
| `<\|pad\|>` | Padding |
| `<\|im_start\|>` | Start of a turn |
| `<\|im_end\|>` | End of a turn |
| `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`, `<\|tool\|>` | Role, one id each |

A turn is `<|im_start|><|role|>\n{content}<|im_end|>\n`. Supervised positions are the assistant's content, the `<|im_end|>` that closes it, and the final `<|eos|>` when the transcript ends on an assistant turn. A generation prompt (`<|im_start|><|assistant|>\n`) is not supervised.

## Corpus

`python/fetch_tokenizer_corpus.py` writes the leading slice of each public stream. That order is reproducible for a dataset revision. The raw files live under `data/tokenizer/` and are gitignored.

| Slice | Dataset | Revision | Train bytes |
| --- | --- | ---: | ---: |
| FineWeb-Edu sample-10BT | `HuggingFaceFW/fineweb-edu` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | 80,001,202 |
| Cosmopedia-v2 | `HuggingFaceTB/smollm-corpus` | `3ba9d605774198c5868892d7a8deda78031a781f` | 32,002,405 |
| TinyStories | `roneneldan/TinyStories` | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` | 32,000,578 |
| English Wikipedia | `KeremTT/FABLE-5.2-MAX-DATABASE` `raw_text_corpus`, `language=en` | `d0bcd6b2181f5588afa9e136a383397dff956620` | 16,001,822 |
| FABLE code | `code_generation`, English | same FABLE revision | 10,001,910 |
| FABLE SQL | `sql_and_data`, English | same | 8,000,051 |
| FABLE JSON | `structured_output`, English | same | 8,000,594 |
| FABLE dialogue | `multi_turn_dialogue`, English | same | 8,003,987 |
| FABLE tools | `tool_calling`, English | same | 8,000,865 |

About 202 MB of training text. Turkish rows are skipped. Each source also keeps the next ~200 KB as a held-out file that was not in the training slice.

```bash
python python/fetch_tokenizer_corpus.py
cargo run -p loopmind -- tokenize-train --config configs/tokenizer.toml
cargo run -p loopmind -- tokenize-stats --heldout data/tokenizer/heldout
```

## Bytes per token

Higher means fewer tokens for the same UTF-8 text. Held-out slices are about 200 KB each. MiniMind is `jingyaogong/MiniMind2` `tokenizer.json`, also vocab 6400, scored on the same files.

| Held-out source | LoopMind | MiniMind |
| --- | ---: | ---: |
| FineWeb-Edu | 3.790 | 3.309 |
| Cosmopedia-v2 | 4.107 | 3.581 |
| TinyStories | 3.688 | 2.961 |
| FABLE English Wikipedia | 3.213 | 2.722 |
| FABLE dialogue | 3.740 | 3.328 |
| FABLE code | 3.207 | 2.787 |
| FABLE SQL | 3.170 | 2.114 |
| FABLE JSON | 2.677 | 2.210 |
| FABLE tools | 3.379 | 2.748 |

LoopMind is denser on every slice. The largest gap is SQL and TinyStories. MiniMind's vocab also covers Chinese, so less of its 6400 ids are spent on this English mix.
