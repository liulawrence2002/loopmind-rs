"""Stream dataset rows as JSON lines for `loopmind prepare-pretrain` and `prepare-sft`.

The stream order is the dataset revision's natural order. `--skip N` drops the
first N emitted rows so a Rust run can resume. Nothing is written except stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

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


def stream_sft(revision: str, skip: int) -> dict:
    """FABLE's English-tagged chat configs. Returns the skip state for later sources."""
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
    return state


MINIMIND_REPO = "jingyaogong/minimind_dataset"
MINIMIND_LICENSE = "apache-2.0 / cc-by-nc-2.0 (minimind_dataset)"
# FABLE's tool-calling system prompt, so MiniMind tool rows read the same way.
TOOL_PROMPT = (
    "You are a helpful assistant with access to the tools listed below. Call a tool only "
    "when it is needed to answer the user. To call one, reply with a JSON object inside "
    "<tool_call></tool_call> tags, in the form {\"name\": <tool name>, \"arguments\": "
    "<arguments object>}. Tool results are returned to you inside "
    "<tool_response></tool_response> tags."
)
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")


def minimind_path(path: str, revision: str) -> Path:
    """The local MiniMind file, downloaded at `revision` when it is missing."""
    local = Path(path)
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download

    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {MINIMIND_REPO}/{local.name} at {revision}", file=sys.stderr)
    return Path(
        hf_hub_download(
            MINIMIND_REPO,
            local.name,
            repo_type="dataset",
            revision=revision,
            local_dir=str(local.parent),
        )
    )


def as_json(value):
    """MiniMind stores `tools`, `tool_calls`, and call arguments as JSON strings."""
    if isinstance(value, str):
        return json.loads(value) if value.strip() else None
    return value


def minimind_messages(conversation: list) -> list | None:
    """MiniMind turns in FABLE's form: tools in the system turn, calls and results in
    <tool_call> and <tool_response> tags. `reasoning_content` is dropped. None when the
    row cannot be rendered."""
    messages = []
    for turn in conversation:
        role = turn.get("role") or ""
        content = (turn.get("content") or "").strip()
        tools = as_json(turn.get("tools"))
        if tools:
            tools = [
                {"type": "function", **tool} if "function" in tool and "type" not in tool else tool
                for tool in tools
            ]
            listing = json.dumps(tools, indent=2, ensure_ascii=False)
            content = f"{content or TOOL_PROMPT}\n\n<tools>\n{listing}\n</tools>"
        calls = as_json(turn.get("tool_calls"))
        if calls:
            rendered = []
            for call in calls:
                function = call.get("function", call)
                arguments = as_json(function.get("arguments")) or {}
                body = json.dumps({"name": function["name"], "arguments": arguments}, ensure_ascii=False)
                rendered.append(f"<tool_call>\n{body}\n</tool_call>")
            content = "\n".join(([content] if content else []) + rendered)
        if role == "tool" and not content.startswith("<tool_response>"):
            content = f"<tool_response>\n{content}\n</tool_response>"
        if role == "assistant" and not content:
            return None
        messages.append({"role": role, "content": content})
    return messages


def stream_minimind(path: str, revision: str, state: dict) -> None:
    """MiniMind SFT rows after FABLE's. Rows with any CJK text are tagged `zh`, so the
    Rust side drops them as non-English. Rows it cannot render go out empty."""
    with open(minimind_path(path, revision), encoding="utf-8") as lines:
        for number, line in enumerate(lines):
            try:
                messages = minimind_messages(json.loads(line).get("conversations") or [])
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                messages = None
            text = "".join(message["content"] for message in messages or [])
            emit(
                {
                    "id": f"minimind-{number:07d}",
                    "config": "minimind_sft",
                    "license": MINIMIND_LICENSE,
                    "language": "zh" if CJK.search(text) else "en",
                    "messages": messages or [],
                },
                state,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["pretrain", "sft"], required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--minimind-file", default="")
    parser.add_argument("--minimind-revision", default="")
    args = parser.parse_args()
    if args.kind == "pretrain":
        if args.source not in PRETRAIN:
            raise SystemExit(f"unknown pretrain source {args.source}")
        stream_pretrain(args.source, args.revision, args.skip)
    else:
        state = stream_sft(args.revision, args.skip)
        if args.minimind_file:
            stream_minimind(args.minimind_file, args.minimind_revision, state)
        sys.stdout.flush()


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
