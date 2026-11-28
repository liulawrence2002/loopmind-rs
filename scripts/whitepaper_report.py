"""Collect the whitepaper results into runs/whitepaper/results.md (and CSVs next to it).

Reads whatever exists so far: runs/bench/*.json from `loopmind bench`, runs/<run>/metrics.jsonl
from training, runs/eval/wp-sft-*.json from `loopmind eval`, and runs/whitepaper/choices.env.
scripts/whitepaper.sh runs it after each stage; it can also be run by hand at any time.

    python3 scripts/whitepaper_report.py
"""

import csv
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WP = ROOT / "runs" / "whitepaper"
BENCH = ROOT / "runs" / "bench"
BIN = ROOT / "target" / "release" / "loopmind"

# name, run dir, bench stage, model config, how it loops in training
COMPARISON = [
    ("LoopMind, loop mean 4", "wp-loop4", "configs/default.toml", "Poisson mean 4, max 16"),
    ("LoopMind, loop mean 8", "wp-loop8", "configs/whitepaper/model-loop8.toml", "Poisson mean 8, max 32"),
    ("No recurrence (same parameters)", "wp-noloop", "configs/whitepaper/model-noloop.toml", "always 1"),
    ("Unshared 10 layers (same compute as r = 4)", "wp-unshared", "configs/whitepaper/model-unshared.toml", "always 1"),
]


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def choices() -> dict:
    values = {}
    path = WP / "choices.env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
    return values


def params(config: str) -> tuple[str, str]:
    try:
        out = subprocess.run([str(BIN), "params", "--config", str(ROOT / config)],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "?", "?"
    total = re.search(r"^\s*total\s+([\d,]+)", out, re.M)
    active = re.search(r"^\s*active per token\s+([\d,]+)", out, re.M)
    return (total.group(1) if total else "?", active.group(1) if active else "?")


def metrics(run: str) -> list[dict]:
    path = ROOT / "runs" / run / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def last_val(rows: list[dict]) -> dict:
    for row in reversed(rows):
        if row.get("val_loss"):
            return row["val_loss"]
    return {}


def throughput(rows: list[dict]) -> str:
    speeds = [row["tokens_per_sec"] for row in rows if row.get("tokens_per_sec")]
    return f"{sum(speeds) / len(speeds):,.0f}" if speeds else "?"


def loops_of(key: str) -> int:
    return int(key.lstrip("r"))


def pct(value) -> str:
    return "" if value is None else f"{100 * value:.1f}"


def best(report: dict) -> tuple[str, float]:
    key = max(report["mean"], key=lambda k: report["mean"][k])
    return key, report["mean"][key]


def headline(report: dict, task: str, key: str):
    score = report["tasks"].get(task, {}).get(key)
    if score is None:
        return None
    return score["acc"] if report["headline"][task] == "acc" else score["acc_norm"]


def task_table(rows: list[tuple[str, dict, str]]) -> list[str]:
    """Rows of (label, report, loop key): tasks as columns."""
    tasks = sorted({task for _, report, _ in rows for task in report["tasks"]})
    chance = {}
    for _, report, _ in rows:
        chance.update(report["chance"])
    out = ["| Model | r | " + " | ".join(tasks) + " | Mean |",
           "| --- | ---: |" + " ---: |" * (len(tasks) + 1),
           "| *chance* | | " + " | ".join(pct(chance.get(task)) for task in tasks) + " | |"]
    for label, report, key in rows:
        cells = [pct(headline(report, task, key)) for task in tasks]
        out.append(f"| {label} | {loops_of(key)} | " + " | ".join(cells) + f" | **{pct(report['mean'][key])}** |")
    return out


# Static figures for the paper: light surface, the reference categorical order, 2px lines,
# 8px markers with a surface ring, a legend plus end-of-line labels, recessive grid.
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600"]
INK, INK2, GRID, AXIS, SURFACE = "#191919", "#666666", "#e7e7e7", "#c2c2c2", "#ffffff"


def nice_step(span: float) -> float:
    raw = span / 5
    for step in (0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100):
        if step >= raw:
            return step
    return raw


def line_chart(path: Path, title: str, x_label: str, y_label: str, series, refs=(), log2=False, x_fmt=str) -> None:
    """series: (name, [(x, y)]) in fixed slot order; refs: (name, y) dashed reference lines."""
    import math
    width, height, left, right, top, bottom = 680, 380, 60, 170, 64, 52
    xs = [x for _, points in series for x, _ in points]
    ys = [y for _, points in series for _, y in points] + [y for _, y in refs]
    if not xs:
        return
    fx = (lambda x: math.log2(x)) if log2 else (lambda x: x)
    x0, x1 = min(map(fx, xs)), max(map(fx, xs))
    x1 = x1 if x1 > x0 else x0 + 1
    step = nice_step(max(ys) - min(ys) or 1)
    y0 = math.floor((min(ys) - step / 2) / step) * step
    y1 = math.ceil((max(ys) + step / 2) / step) * step
    px = lambda x: left + (fx(x) - x0) / (x1 - x0) * (width - left - right)
    py = lambda y: top + (y1 - y) / (y1 - y0) * (height - top - bottom)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Inter, Helvetica, Arial, sans-serif">',
           f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
           f'<text x="{left}" y="22" font-size="15" font-weight="600" fill="{INK}">{title}</text>']
    y = y0
    while y <= y1 + 1e-9:
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{py(y):.1f}" y2="{py(y):.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{py(y) + 4:.1f}" font-size="11" fill="{INK2}" text-anchor="end">{y:g}</text>')
        y += step
    for x in sorted(set(xs)):
        out.append(f'<text x="{px(x):.1f}" y="{height - bottom + 18}" font-size="11" fill="{INK2}" text-anchor="middle">{x_fmt(x)}</text>')
    out.append(f'<line x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}" stroke="{AXIS}" stroke-width="1"/>')
    out.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 10}" font-size="12" fill="{INK2}" text-anchor="middle">{x_label}</text>')
    out.append(f'<text transform="translate(16 {(top + height - bottom) / 2:.1f}) rotate(-90)" font-size="12" fill="{INK2}" text-anchor="middle">{y_label}</text>')
    labels = []  # (y in px, x in px, text, ink): end-of-line and reference labels, spaced below
    for name, value in refs:
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{py(value):.1f}" y2="{py(value):.1f}" stroke="{INK2}" stroke-width="1.5" stroke-dasharray="5 4"/>')
        labels.append((py(value), width - right + 8, f"{name} {value:.1f}", INK2))
    legend_x = left
    for index, (name, points) in enumerate(series):
        color = SERIES[index % len(SERIES)]
        out.append(f'<line x1="{legend_x}" x2="{legend_x + 18}" y1="42" y2="42" stroke="{color}" stroke-width="2"/>')
        out.append(f'<circle cx="{legend_x + 9}" cy="42" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
        out.append(f'<text x="{legend_x + 24}" y="46" font-size="12" fill="{INK}">{name}</text>')
        legend_x += 34 + 7 * len(name)
        points = sorted(points)
        path_d = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(points))
        out.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
        for x, y in points:
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2"><title>{name}: {y:.1f} at {x_fmt(x)}</title></circle>')
        last_x, last_y = points[-1]
        labels.append((py(last_y), px(last_x) + 10, f"{name} {last_y:.1f}", INK))
    labels.sort()
    placed = []
    for y_px, x_px, text, ink in labels:
        if placed and y_px < placed[-1] + 13:
            y_px = placed[-1] + 13
        placed.append(y_px)
        out.append(f'<text x="{x_px:.1f}" y="{y_px + 4:.1f}" font-size="11" fill="{ink}">{text}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n")


def main() -> None:
    WP.mkdir(parents=True, exist_ok=True)
    picked = choices()
    out = ["# Whitepaper results", "",
           "Generated by `scripts/whitepaper_report.py` from the runs present when it ran. "
           "Benchmarks are zero-shot log-likelihood scores in percent (`acc_norm`, or `acc` for BLiMP "
           "and LAMBADA); docs/whitepaper.md describes the setup.", ""]
    if picked:
        out += ["Pipeline choices: " + ", ".join(f"`{k}={v}`" for k, v in picked.items()), ""]

    out += ["## Models", "", "| Model | Total params | Active params | Loops in training |",
            "| --- | ---: | ---: | --- |"]
    for label, _, config, loops in COMPARISON:
        total, active = params(config)
        out.append(f"| {label} | {total} | {active} | {loops} |")
    out.append("")

    loop4_run = Path(picked.get("LOOP4_RUN", "runs/wp-loop4")).name
    runs50 = [(label, loop4_run if stage == "wp-loop4" else stage, stage) for label, stage, _, _ in COMPARISON]
    out += ["## 50M tokens: validation loss and training speed", "",
            "Masked next-token loss on 64 held-out pretrain sequences at the end of training.", "",
            "| Model | r = 1 | r = 2 | r = 4 | r = 8 | r = 16 | r = 32 | tokens/s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, run, _ in runs50:
        rows = metrics(run)
        if not rows:
            continue
        val = last_val(rows)
        cells = [f"{val[f'r{r}']:.3f}" if f"r{r}" in val else "" for r in (1, 2, 4, 8, 16, 32)]
        out.append(f"| {label} | " + " | ".join(cells) + f" | {throughput(rows)} |")
    out.append("")

    reports = {stage: read_json(BENCH / f"{stage}.json") for _, _, stage in runs50}
    have = [(label, reports[stage]) for label, _, stage in runs50 if reports[stage]]
    if have:
        out += ["## 50M tokens: benchmarks at each model's best loop count", ""]
        out += task_table([(label, report, best(report)[0]) for label, report in have])
        out += ["", "## Test-time loops: benchmark mean by r", "",
                "| Model | " + " | ".join(f"r = {r}" for r in (1, 2, 4, 8, 16, 32)) + " |",
                "| --- |" + " ---: |" * 6]
        scaling = []
        for label, report in have:
            cells = [pct(report["mean"].get(f"r{r}")) for r in (1, 2, 4, 8, 16, 32)]
            out.append(f"| {label} | " + " | ".join(cells) + " |")
            for key, value in report["mean"].items():
                scaling.append({"model": label, "loops": loops_of(key), "mean": value})
        looped = [(label.replace("LoopMind, ", ""), [(loops_of(k), 100 * v) for k, v in report["mean"].items()])
                  for label, report in have if len(report["mean"]) > 1]
        refs = [(label.split(" (")[0], 100 * report["mean"]["r1"]) for label, report in have
                if len(report["mean"]) == 1 and "r1" in report["mean"]]
        if looped:
            line_chart(WP / "fig-scaling-50m.svg", "Benchmark mean by loop count at inference, 50M tokens",
                       "loops at inference (r)", "mean accuracy, %", looped, refs, log2=True)
            out += ["![Benchmark mean by r at 50M tokens](fig-scaling-50m.svg)", ""]
        with open(WP / "scaling-50m.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, ["model", "loops", "mean"])
            writer.writeheader()
            writer.writerows(scaling)
        out.append("")

    big = [("LoopMind, 500M tokens", "pretrain-500m", "wp-500m-loop"),
           ("No recurrence, 500M tokens", "wp-500m-noloop", "wp-500m-noloop")]
    big_reports = [(label, run, read_json(BENCH / f"{stage}.json")) for label, run, stage in big]
    if any(report for _, _, report in big_reports):
        out += ["## 500M tokens", ""]
        rows = [(label, report, best(report)[0]) for label, _, report in big_reports if report]
        out += task_table(rows)
        loop_report = big_reports[0][2]
        base_report = big_reports[1][2]
        if loop_report and len(loop_report["mean"]) > 1:
            refs = [("No recurrence", 100 * base_report["mean"]["r1"])] if base_report and "r1" in base_report["mean"] else []
            line_chart(WP / "fig-scaling-500m.svg", "Benchmark mean by loop count at inference, 500M tokens",
                       "loops at inference (r)", "mean accuracy, %",
                       [("LoopMind", [(loops_of(k), 100 * v) for k, v in loop_report["mean"].items()])], refs, log2=True)
            out += ["", "![Benchmark mean by r at 500M tokens](fig-scaling-500m.svg)"]
        if loop_report:
            out += ["", "LoopMind at 500M tokens by r: " + ", ".join(
                f"r = {loops_of(k)}: {pct(v)}" for k, v in sorted(loop_report["mean"].items(), key=lambda kv: loops_of(kv[0])))]
        out += ["", "| Model | Final val loss by r | tokens/s |", "| --- | --- | ---: |"]
        for label, run, _ in big_reports:
            rows = metrics(run)
            if rows:
                val = last_val(rows)
                text = ", ".join(f"r{loops_of(k)} {v:.3f}" for k, v in sorted(val.items(), key=lambda kv: loops_of(kv[0])))
                out.append(f"| {label} | {text} | {throughput(rows)} |")
        out.append("")

    curve = []
    for path in sorted(BENCH.glob("curve-*.json")):
        match = re.match(r"curve-(loop|noloop)-(\d+)", path.stem)
        report = read_json(path)
        if match and report:
            for key, value in report["mean"].items():
                curve.append({"model": match.group(1), "step": int(match.group(2)),
                              "tokens": int(match.group(2)) * 32768, "loops": loops_of(key), "mean": value})
    if curve:
        curve.sort(key=lambda row: (row["model"], row["loops"], row["step"]))
        with open(WP / "curve-500m.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, ["model", "step", "tokens", "loops", "mean"])
            writer.writeheader()
            writer.writerows(curve)
        series = sorted({(row["model"], row["loops"]) for row in curve}, key=lambda s: (s[0] != "loop", s[1]))
        line_chart(WP / "fig-curve-500m.svg", "Benchmark mean while training on 500M tokens (1,000 items per task)",
                   "tokens seen", "mean accuracy, %",
                   [(f"{'LoopMind' if m == 'loop' else 'No recurrence'} r = {r}",
                     [(row["tokens"], 100 * row["mean"]) for row in curve if row["model"] == m and row["loops"] == r])
                    for m, r in series], x_fmt=lambda x: f"{x / 1e6:.0f}M")
        steps = sorted({row["step"] for row in curve})
        out += ["## Learning curves (1,000 items per task)", "", "![Learning curves](fig-curve-500m.svg)", "",
                "| Tokens seen | " + " | ".join(f"{m} r = {r}" for m, r in series) + " |",
                "| ---: |" + " ---: |" * len(series)]
        for step in steps:
            cells = []
            for model, loops in series:
                hit = [row["mean"] for row in curve if row["model"] == model and row["loops"] == loops and row["step"] == step]
                cells.append(pct(hit[0]) if hit else "")
            out.append(f"| {step * 32768 / 1e6:.0f}M | " + " | ".join(cells) + " |")
        out.append("")

    pairs = [("LoopMind, loop mean 4", "wp-loop4", "wp-loop4-s7"), ("No recurrence", "wp-noloop", "wp-noloop-s7")]
    seed_rows = []
    for label, first, second in pairs:
        a, b = read_json(BENCH / f"{first}.json"), read_json(BENCH / f"{second}.json")
        if a and b:
            (ka, va), (kb, vb) = best(a), best(b)
            seed_rows.append(f"| {label} | {pct(va)} (r = {loops_of(ka)}) | {pct(vb)} (r = {loops_of(kb)}) | {100 * abs(va - vb):.1f} |")
    if seed_rows:
        out += ["## Seed spread at 50M tokens", "", "| Model | Seed 42 | Seed 7 | Difference |",
                "| --- | ---: | ---: | ---: |"] + seed_rows + [""]

    sft = [("LoopMind + SFT", "wp-sft-loop"), ("No recurrence + SFT", "wp-sft-noloop")]
    sft_rows = []
    for label, stage in sft:
        evaluation = read_json(ROOT / "runs" / "eval" / f"{stage}.json")
        bench = read_json(BENCH / f"{stage}.json")
        if evaluation or bench:
            sft_rows.append((label, evaluation, bench))
    if sft_rows:
        out += ["## After SFT", "", "| Model | FABLE eval loss (best r) | FABLE answers, macro | Benchmark mean (best r) |",
                "| --- | ---: | ---: | ---: |"]
        for label, evaluation, bench in sft_rows:
            loss = macro = mean = ""
            if evaluation:
                key = min(evaluation["nll"], key=lambda k: evaluation["nll"][k])
                loss = f"{evaluation['nll'][key]:.3f} (r = {loops_of(key)})"
                macro = f"{100 * evaluation['recommended']['macro_accuracy']:.1f}% (r = {evaluation['recommended']['loops']})"
            if bench:
                key, value = best(bench)
                mean = f"{pct(value)} (r = {loops_of(key)})"
            out.append(f"| {label} | {loss} | {macro} | {mean} |")
        for label, evaluation, _ in sft_rows:
            if evaluation and evaluation.get("smoke"):
                out += ["", f"### {label}: chat samples", ""]
                for sample in evaluation["smoke"][:4]:
                    prompt = sample.get("prompt", "")
                    answer = sample.get("answer", sample.get("text", ""))
                    out += [f"**{prompt}**", "", "```text", str(answer)[:600], "```", ""]
        out.append("")

    (WP / "results.md").write_text("\n".join(out) + "\n")
    print(WP / "results.md")


if __name__ == "__main__":
    main()
