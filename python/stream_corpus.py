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
    "simplewiki": ("wikimedia/wikipedia", "20231101.simple", "text"),
}

# FABLE configs the eval rows are drawn from. Prompts from these are kept aside so the
# other sources can drop any row that repeats one.
EVAL_CONFIGS = {"math_reasoning", "logic_reasoning", "science_knowledge", "structured_output"}

# English-only check, shared by every SFT source. It judges the answers. A row is not
# English when an assistant answer is mostly another script (over 30% CJK, or over 30%
# Greek/Cyrillic/Arabic/Hebrew/Indic/Thai/Georgian), when an answer of 15+ words is
# mostly Spanish/French/German/Italian/Portuguese/Dutch (their function words are over
# 15% of its words, at least 4 different ones, and outnumber English ones), or when the
# conversation as a whole is over 2% CJK. A name like 秦始皇 or a π inside an English
# answer, a list of Dutch names, and JSON all pass. The 2% line drops Chinese chats with
# short answers and MiniMind's English rows whose tool definitions are in Chinese.
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")
OTHER_SCRIPT = re.compile(
    r"[\u0370-\u03ff\u0400-\u052f\u0590-\u05ff\u0600-\u06ff\u0900-\u0dff\u0e00-\u0e7f\u10a0-\u10ff]"
)
LETTER = re.compile(r"[^\W\d_]")
WORD = re.compile(r"[a-zà-öø-ÿ]+", re.IGNORECASE)
ENGLISH_WORDS = set(
    """the of and to in is that it for you with as on are this be was by not or at from have
    but they your can an we which their will has if more what about there would been one all
    when so how its our also into than other some these may such only them do does could
    should those i my me""".split()
)
FOREIGN_WORDS = set(
    """el la los las del que por para una uno es está están con pero como más muy también
    le les des du et est pour dans avec une sont qui pas sur ce cette mais très aussi être
    der die das und ist nicht mit ein eine einen dem den zu von auf für sich auch werden wird
    oder il di che della delle degli sono questo questa anche essere gli os uma não com são
    isso ele ela mais também ser het een van niet zijn ook voor met op dat""".split()
) - ENGLISH_WORDS


def foreign_answer(text: str) -> bool:
    words = [word.lower() for word in WORD.findall(text)]
    if len(words) < 15:
        return False
    english = sum(word in ENGLISH_WORDS for word in words)
    foreign = [word for word in words if word in FOREIGN_WORDS]
    return (
        len(foreign) > english
        and len(foreign) >= 0.15 * len(words)
        and len(set(foreign)) >= 4
    )


def share(pattern: re.Pattern, text: str) -> float:
    """Characters matching `pattern` as a share of the letters in `text`."""
    letters = len(LETTER.findall(text))
    return len(pattern.findall(text)) / letters if letters else 0.0


def language_of(messages: list) -> str:
    """`en`, or why the answers are not English."""
    for message in messages:
        if message["role"] != "assistant":
            continue
        answer = message["content"]
        if share(CJK, answer) > 0.3:
            return "not-en:cjk"
        if share(OTHER_SCRIPT, answer) > 0.3:
            return "not-en:script"
        if foreign_answer(answer):
            return "not-en:foreign-words"
    if share(CJK, "".join(message["content"] for message in messages)) > 0.02:
        return "not-en:cjk-conversation"
    return "en"


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def repeats_eval(messages: list, eval_prompts: set) -> bool:
    return any(m["role"] == "user" and normalize(m["content"]) in eval_prompts for m in messages)


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


def stream_sft(revision: str, skip: int) -> tuple[dict, set]:
    """FABLE's chat configs. A row stays English only if FABLE tags it `en` and it passes
    `language_of`. Returns the skip state for later sources and the normalized user
    prompts of the eval configs."""
    state = {"left": skip, "written": 0}
    eval_prompts = set()
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
            if config in EVAL_CONFIGS:
                eval_prompts.update(
                    normalize(m["content"]) for m in messages if m["role"] == "user"
                )
            language = row.get("language") or ""
            if language.strip().lower() in ("en", "english") and language_of(messages) != "en":
                language = language_of(messages)
            emit(
                {
                    "id": str(row.get("id") or ""),
                    "config": config,
                    "category": row.get("category") or config,
                    "license": row.get("license") or "",
                    "language": language,
                    "messages": messages,
                },
                state,
            )
    sys.stdout.flush()
    return state, eval_prompts


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


def stream_minimind(path: str, revision: str, state: dict, eval_prompts: set) -> None:
    """MiniMind SFT rows after FABLE's, tagged by `language_of` so the Rust side drops the
    non-English ones (most are Chinese). Rows it cannot render go out empty."""
    with open(minimind_path(path, revision), encoding="utf-8") as lines:
        for number, line in enumerate(lines):
            try:
                messages = minimind_messages(json.loads(line).get("conversations") or [])
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                messages = None
            messages = messages or []
            emit(
                {
                    "id": f"minimind-{number:07d}",
                    "config": "minimind_sft",
                    "license": MINIMIND_LICENSE,
                    "language": language_of(messages),
                    "messages": messages,
                    "trim_to_fit": True,
                    "eval_overlap": repeats_eval(messages, eval_prompts),
                },
                state,
            )


SMOLTALK_REPO = "HuggingFaceTB/smol-smoltalk"


def stream_smoltalk(revision: str, state: dict, eval_prompts: set) -> None:
    """The train split of smol-smoltalk (Apache-2.0), the SFT mix for SmolLM2-135M/360M.
    The parquet files are fetched once into the Hugging Face cache. Config names keep the
    row's source, as `smoltalk:<source>`."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            SMOLTALK_REPO, repo_type="dataset", revision=revision, allow_patterns=["data/train-*"]
        )
    )
    number = 0
    for path in sorted(root.glob("data/train-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["messages", "source"]):
            for row in batch.to_pylist():
                messages = [
                    {"role": m.get("role") or "", "content": m.get("content") or ""}
                    for m in row.get("messages") or []
                ]
                emit(
                    {
                        "id": f"smoltalk-{number:07d}",
                        "config": f"smoltalk:{row.get('source') or 'unknown'}",
                        "license": "apache-2.0",
                        "language": language_of(messages),
                        "messages": messages,
                        "trim_to_fit": True,
                        "eval_overlap": repeats_eval(messages, eval_prompts),
                    },
                    state,
                )
                number += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["pretrain", "sft"], required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--minimind-file", default="")
    parser.add_argument("--minimind-revision", default="")
    parser.add_argument("--smoltalk-revision", default="")
    args = parser.parse_args()
    if args.kind == "pretrain":
        if args.source not in PRETRAIN:
            raise SystemExit(f"unknown pretrain source {args.source}")
        stream_pretrain(args.source, args.revision, args.skip)
    else:
        state, eval_prompts = stream_sft(args.revision, args.skip)
        if args.minimind_file:
            stream_minimind(args.minimind_file, args.minimind_revision, state, eval_prompts)
        if args.smoltalk_revision:
            stream_smoltalk(args.smoltalk_revision, state, eval_prompts)
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
