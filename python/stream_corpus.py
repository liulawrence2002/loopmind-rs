"""Stream dataset rows as JSON lines for `loopmind prepare-pretrain` and `prepare-sft`.

The stream order is the dataset revision's natural order. `--skip N` drops the
first N emitted rows so a Rust run can resume. Nothing is written except stdout.
"""

from __future__ import annotations

import argparse
import json
import sys

from datasets import load_dataset

PRETRAIN = {
    "fineweb": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text"),
    "tinystories": ("roneneldan/TinyStories", None, "text"),
    "fable-wiki": ("KeremTT/FABLE-5.2-MAX-DATABASE", "raw_text_corpus", "wiki"),
}

SFT_CONFIGS = [
    "general_instruction",
    "multi_turn_dialogue",
    "code_generation",
    "sql_and_data",
    "math_reasoning",
    "logic_reasoning",
    "reading_comprehension",
    "science_knowledge",
    "tool_calling",
    "structured_output",
]


def emit(obj: dict, skip_state: dict) -> None:
    if skip_state["left"] > 0:
        skip_state["left"] -= 1
        return
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    skip_state["written"] += 1
    if skip_state["written"] % 64 == 0:
        sys.stdout.flush()


def pretrain_text(row: dict, kind: str) -> str | None:
    if kind == "wiki" and str(row.get("language", "")).lower() != "en":
        return None
    text = str(row.get("text") or "").strip()
    return text or None


def stream_pretrain(source: str, revision: str, skip: int) -> None:
    dataset, config, kind = PRETRAIN[source]
    kwargs = {"path": dataset, "split": "train", "streaming": True, "revision": revision}
    if config is not None:
        kwargs["name"] = config
    state = {"left": skip, "written": 0}
    for row in load_dataset(**kwargs):
        text = pretrain_text(row, kind)
        if text is None:
            continue
        emit({"text": text}, state)
    sys.stdout.flush()


def stream_sft(revision: str, skip: int) -> None:
    state = {"left": skip, "written": 0}
    for config in SFT_CONFIGS:
        rows = load_dataset(
            "KeremTT/FABLE-5.2-MAX-DATABASE",
            config,
            split="train",
            streaming=True,
            revision=revision,
        )
        for row in rows:
            messages = []
            for message in row.get("messages") or []:
                messages.append(
                    {
                        "role": message.get("role") or "",
                        "content": message.get("content") or "",
                    }
                )
            emit(
                {
                    "id": str(row.get("id") or ""),
                    "config": config,
                    "category": row.get("category") or config,
                    "license": row.get("license") or "",
                    "language": row.get("language") or "",
                    "messages": messages,
                },
                state,
            )
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["pretrain", "sft"], required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()
    if args.kind == "pretrain":
        if args.source not in PRETRAIN:
            raise SystemExit(f"unknown pretrain source {args.source}")
        stream_pretrain(args.source, args.revision, args.skip)
    else:
        stream_sft(args.revision, args.skip)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # The Rust side closes the pipe once a source hits its token budget.
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
        raise SystemExit(0)
