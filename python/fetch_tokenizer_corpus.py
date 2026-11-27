"""Download a reproducible tokenizer corpus.

The sample is the leading slice of each public dataset stream, so the same
dataset revision yields the same files. Bytes follow the pretrain mix
(FineWeb-Edu 50%, Cosmopedia-v2 20%, TinyStories 20%, English Wikipedia 10%)
plus English FABLE chat rows that contain code, SQL, JSON, and dialogue.

Nothing under ``data/`` is committed. Train with::

    python python/fetch_tokenizer_corpus.py
    cargo run -p loopmind -- tokenize-train --config configs/tokenizer.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import dataset_info

# name, dataset, config, train bytes, held-out bytes, how to read a row
SOURCES = [
    ("fineweb", "HuggingFaceFW/fineweb-edu", "sample-10BT", 80_000_000, 200_000, "text"),
    ("cosmopedia", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 32_000_000, 200_000, "text"),
    ("tinystories", "roneneldan/TinyStories", None, 32_000_000, 200_000, "text"),
    ("fable-wiki", "KeremTT/FABLE-5.2-MAX-DATABASE", "raw_text_corpus", 16_000_000, 200_000, "wiki"),
    ("fable-code", "KeremTT/FABLE-5.2-MAX-DATABASE", "code_generation", 10_000_000, 200_000, "chat"),
    ("fable-sql", "KeremTT/FABLE-5.2-MAX-DATABASE", "sql_and_data", 8_000_000, 200_000, "chat"),
    ("fable-json", "KeremTT/FABLE-5.2-MAX-DATABASE", "structured_output", 8_000_000, 200_000, "chat"),
    ("fable-dialogue", "KeremTT/FABLE-5.2-MAX-DATABASE", "multi_turn_dialogue", 8_000_000, 200_000, "chat"),
    ("fable-tools", "KeremTT/FABLE-5.2-MAX-DATABASE", "tool_calling", 8_000_000, 200_000, "chat"),
]


def row_text(row: dict, kind: str) -> str | None:
    language = str(row.get("language", "en")).lower()
    if language not in {"en", "english", ""}:
        return None
    if kind == "text":
        text = row.get("text") or ""
    elif kind == "wiki":
        if language != "en":
            return None
        text = row.get("text") or ""
    else:
        if language != "en":
            return None
        parts = []
        for message in row.get("messages") or []:
            role = message.get("role", "")
            content = message.get("content") or ""
            parts.append(f"{role}: {content}")
        text = "\n".join(parts)
    text = str(text).strip()
    return text or None


def write_slice(name: str, dataset: str, config: str | None, train_bytes: int, held_bytes: int, kind: str, root: Path) -> None:
    sha = dataset_info(dataset).sha
    print(f"{name}: {dataset} config={config} revision={sha}", flush=True)
    kwargs = {"path": dataset, "split": "train", "streaming": True}
    if config is not None:
        kwargs["name"] = config
    stream = load_dataset(**kwargs)
    train_path = root / "train" / f"{name}.txt"
    held_path = root / "heldout" / f"{name}.txt"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    held_path.parent.mkdir(parents=True, exist_ok=True)
    train_got = 0
    held_got = 0
    with train_path.open("w", encoding="utf-8") as train, held_path.open("w", encoding="utf-8") as held:
        for row in stream:
            text = row_text(row, kind)
            if text is None:
                continue
            blob = (text + "\n\n").encode("utf-8")
            if train_got < train_bytes:
                train.write(blob.decode("utf-8"))
                train_got += len(blob)
            elif held_got < held_bytes:
                held.write(blob.decode("utf-8"))
                held_got += len(blob)
            else:
                break
    print(f"  train {train_got} bytes, heldout {held_got} bytes", flush=True)
    if train_got < train_bytes // 2:
        raise SystemExit(f"{name} produced only {train_got} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/tokenizer"))
    args = parser.parse_args()
    for source in SOURCES:
        write_slice(*source, args.out)
    print(f"wrote corpus under {args.out}")


if __name__ == "__main__":
    main()
