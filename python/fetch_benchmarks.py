"""Download zero-shot benchmarks into data/bench/<task>.jsonl for `loopmind bench`.

Each line is {"id", "task", "context", "choices", "label"}. The model scores every choice
as a continuation of the context, by log-likelihood, as lm-evaluation-harness does; the
prompt formats follow its task definitions. A task with one choice (LAMBADA) is scored by
whether greedy decoding reproduces it. A BLiMP item has an empty context and the
grammatical sentence as choice 0.

Revisions are pinned so every run scores the same items.

    python python/fetch_benchmarks.py --out data/bench
"""

import argparse
import json
import re
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset

REVISIONS = {
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "baber/piqa": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "allenai/sciq": "2c94ad3e1aafab77146f384e23536f97a4849815",
    "allenai/openbookqa": "388097ea7776314e93a529163e0fea805b8a6454",
    "EleutherAI/lambada_openai": "900124bf3b8235c6daf21033af9948b3f07346c4",
    "nyu-mll/blimp": "877fba0801ffb7cbd8c39c1ff314a46f053f6036",
}


def load(name: str, config: str | None, split: str):
    kwargs = {"path": name, "split": split, "revision": REVISIONS[name]}
    if config is not None:
        kwargs["name"] = config
    return load_dataset(**kwargs)


def arc(config: str, task: str):
    for row in load("allenai/ai2_arc", config, "test"):
        labels = row["choices"]["label"]
        if row["answerKey"] not in labels:
            continue
        yield {
            "id": row["id"],
            "task": task,
            "context": f"Question: {row['question']}\nAnswer:",
            "choices": [" " + text for text in row["choices"]["text"]],
            "label": labels.index(row["answerKey"]),
        }


def piqa():
    for index, row in enumerate(load("baber/piqa", None, "validation")):
        yield {
            "id": f"piqa-{index}",
            "task": "piqa",
            "context": f"Question: {row['goal']}\nAnswer:",
            "choices": [" " + row["sol1"], " " + row["sol2"]],
            "label": int(row["label"]),
        }


def hellaswag_text(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def hellaswag():
    for row in load("Rowan/hellaswag", None, "validation"):
        context = row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize()
        yield {
            "id": f"hellaswag-{row['ind']}",
            "task": "hellaswag",
            "context": hellaswag_text(context),
            "choices": [" " + hellaswag_text(ending) for ending in row["endings"]],
            "label": int(row["label"]),
        }


def sciq():
    for index, row in enumerate(load("allenai/sciq", None, "test")):
        choices = [row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"]]
        yield {
            "id": f"sciq-{index}",
            "task": "sciq",
            "context": f"{row['support'].lstrip()}\nQuestion: {row['question']}\nAnswer:",
            "choices": [" " + choice for choice in choices],
            "label": 3,
        }


def openbookqa():
    for row in load("allenai/openbookqa", "main", "test"):
        labels = row["choices"]["label"]
        yield {
            "id": row["id"],
            "task": "openbookqa",
            "context": row["question_stem"],
            "choices": [" " + text for text in row["choices"]["text"]],
            "label": labels.index(row["answerKey"].strip()),
        }


def lambada():
    for index, row in enumerate(load("EleutherAI/lambada_openai", "en", "test")):
        context, word = row["text"].rsplit(" ", 1)
        yield {
            "id": f"lambada-{index}",
            "task": "lambada",
            "context": context,
            "choices": [" " + word],
            "label": 0,
        }


def blimp():
    configs = get_dataset_config_names("nyu-mll/blimp", revision=REVISIONS["nyu-mll/blimp"])
    for config in sorted(configs):
        for index, row in enumerate(load("nyu-mll/blimp", config, "train")):
            yield {
                "id": f"blimp-{config}-{index}",
                "task": "blimp",
                "context": "",
                "choices": [row["sentence_good"], row["sentence_bad"]],
                "label": 0,
            }


TASKS = {
    "arc_easy": lambda: arc("ARC-Easy", "arc_easy"),
    "arc_challenge": lambda: arc("ARC-Challenge", "arc_challenge"),
    "piqa": piqa,
    "hellaswag": hellaswag,
    "sciq": sciq,
    "openbookqa": openbookqa,
    "lambada": lambada,
    "blimp": blimp,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/bench")
    parser.add_argument("--tasks", default=",".join(TASKS))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for task in args.tasks.split(","):
        rows = list(TASKS[task]())
        with open(out / f"{task}.jsonl", "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[task] = len(rows)
        print(f"{task}: {len(rows)} items")
    manifest = {"revisions": REVISIONS, "items": counts}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
