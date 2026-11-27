#!/usr/bin/env bash
# After M5 SFT: evaluate the model, tune the loop dial, run a low-LR SFT pass and
# self-generated DPO, and keep whichever checkpoint wins. Writes runs/best.json, which
# `loopmind chat` uses by default, and runs/eval/summary.md.
#
#   WAIT_PID=<pid> DEVICE=cuda scripts/refine.sh     # start when that process exits
#
# Every stage leaves runs/refine/<stage>.done, so running the script again resumes.
# Training stages retry with --resume up to 3 times. A failed stage is logged and the
# pipeline continues with the checkpoints it has.
#
# Keep rules (held-out eval rows, at the loop count and sampling tuned on the SFT model):
#   refine pass: kept if its masked loss is lower than SFT's.
#   DPO: kept if macro answer accuracy rises by >= 0.5 points and the masked loss
#        rises by <= 0.05.
set -u
cd "$(dirname "$0")/.." || exit 1
BIN=${BIN:-./target/release/loopmind}
DEVICE=${DEVICE:-cuda}
STATE=runs/refine
mkdir -p "$STATE" runs/eval
LOG=$STATE/pipeline.log

log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
done_mark() { touch "$STATE/$1.done"; }
is_done() { [ -e "$STATE/$1.done" ]; }

if [ -n "${WAIT_PID:-}" ]; then
  log "waiting for process $WAIT_PID (SFT) to exit"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
if ! grep -q "supervisor: finished\|^finished [0-9]* sft steps" runs/sft/train.log 2>/dev/null; then
  log "runs/sft did not finish; stopping"
  exit 1
fi

# train_stage NAME RUN_DIR COMMAND...: run, and on failure resume from the latest checkpoint.
train_stage() {
  local name=$1 dir=$2
  shift 2
  mkdir -p "$dir"
  for attempt in 0 1 2 3; do
    # Resume whenever a checkpoint exists, after trimming the log back to it.
    local extra=()
    if [ -e "$dir/latest.json" ]; then
      python3 - "$dir" <<'PY'
import json, os, sys
run = sys.argv[1]
step = json.load(open(f"{run}/latest.json"))["step"]
path = f"{run}/metrics.jsonl"
if os.path.exists(path):
    rows = [line for line in open(path) if json.loads(line)["step"] <= step]
    open(path, "w").writelines(rows)
PY
      extra=(--resume)
    fi
    log "$name attempt $attempt ${extra[*]}"
    "$@" "${extra[@]}" >> "$dir/train.log" 2>&1
    if tail -3 "$dir/train.log" | grep -q "^finished"; then
      done_mark "$name"
      return 0
    fi
    sleep 30
  done
  log "$name failed after retries"
  return 1
}

# field REPORT EXPR: a value from an eval report, via Python.
field() {
  python3 - "$1" "$2" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
print(eval(sys.argv[2], {"r": report}))
PY
}

# 1. Start the model: masked loss at r = 1..16, greedy answers at r = 1..16, sampling at the best r.
if ! is_done eval-sft; then
  log "eval runs/sft"
  if "$BIN" eval --checkpoint runs/sft --out runs/eval/sft.json --device "$DEVICE" >> "$LOG" 2>&1; then
    done_mark eval-sft
  else
    log "eval of runs/sft failed; stopping"
    exit 1
  fi
fi
LOOPS=$(field runs/eval/sft.json 'r["recommended"]["loops"]')
SAMPLING=$(field runs/eval/sft.json 'r["recommended"]["sampling"]["name"]')
log "tuned on SFT: r = $LOOPS, sampling $SAMPLING"

# The same setting is used to compare every later checkpoint.
eval_at_setting() {
  "$BIN" eval --checkpoint "$1" --out "$2" --device "$DEVICE" \
    --gen-loops "$LOOPS" --sampling "$SAMPLING" >> "$LOG" 2>&1
}

write_best() {
  python3 - "$@" <<'PY'
import json, sys
checkpoint, report, reason = sys.argv[1:4]
r = json.load(open(report))
rec = json.load(open("runs/eval/sft.json"))["recommended"]
json.dump({"checkpoint": checkpoint, "loops": rec["loops"], "sampling": rec["sampling"],
           "reason": reason, "report": report}, open("runs/best.json", "w"), indent=2)
PY
}

# Masked loss and macro accuracy of a report at the tuned setting.
score_of() {
  python3 - "$1" "$LOOPS" "$SAMPLING" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); loops = int(sys.argv[2]); name = sys.argv[3]
nll = r["nll"][f"r{loops}"]
acc = next(g["macro_accuracy"] for g in r["generation"]
           if g["loops"] == loops and g["sampling"]["name"] == name)
print(f"{nll:.4f} {acc:.4f}")
PY
}

BEST=runs/sft
BEST_REPORT=runs/eval/sft.json
write_best "$BEST" "$BEST_REPORT" "SFT, tuned r and sampling"

# 2. Low-LR SFT pass from the final SFT checkpoint.
if ! is_done sft-refine; then
  train_stage sft-refine runs/sft-refine "$BIN" sft --config configs/sft-refine.toml --device "$DEVICE"
fi
if is_done sft-refine && ! is_done eval-sft-refine; then
  log "eval runs/sft-refine"
  eval_at_setting runs/sft-refine runs/eval/sft-refine.json && done_mark eval-sft-refine
fi
if is_done eval-sft-refine; then
  read -r SFT_NLL SFT_ACC <<< "$(score_of runs/eval/sft.json)"
  read -r REF_NLL REF_ACC <<< "$(score_of runs/eval/sft-refine.json)"
  if python3 -c "import sys; sys.exit(0 if $REF_NLL < $SFT_NLL else 1)"; then
    BEST=runs/sft-refine
    BEST_REPORT=runs/eval/sft-refine.json
    log "keeping the refine pass: masked loss $REF_NLL < $SFT_NLL"
    write_best "$BEST" "$BEST_REPORT" "refine pass: held-out loss $REF_NLL < SFT $SFT_NLL"
  else
    log "discarding the refine pass: masked loss $REF_NLL >= $SFT_NLL"
  fi
fi

# 3. Self-generated DPO from the best checkpoint so far.
mkdir -p runs/dpo
if ! is_done dpo-pairs; then
  log "dpo pairs from $BEST at r = $LOOPS"
  if "$BIN" dpo-pairs --checkpoint "$BEST" --out runs/dpo/pairs.jsonl --device "$DEVICE" \
      --loops "$LOOPS" --sampling t0.7-k40 >> "$LOG" 2>&1; then
    echo "$BEST" > "$STATE/dpo-init"
    done_mark dpo-pairs
  else
    log "dpo pairs failed"
  fi
fi
if is_done dpo-pairs && ! is_done dpo; then
  train_stage dpo runs/dpo "$BIN" dpo --config configs/dpo.toml --device "$DEVICE" --init "$(cat "$STATE/dpo-init")"
fi
if is_done dpo && ! is_done eval-dpo; then
  log "eval runs/dpo"
  eval_at_setting runs/dpo runs/eval/dpo.json && done_mark eval-dpo
fi
if is_done eval-dpo; then
  read -r BEST_NLL BEST_ACC <<< "$(score_of "$BEST_REPORT")"
  read -r DPO_NLL DPO_ACC <<< "$(score_of runs/eval/dpo.json)"
  if python3 -c "import sys; sys.exit(0 if $DPO_ACC >= $BEST_ACC + 0.005 and $DPO_NLL <= $BEST_NLL + 0.05 else 1)"; then
    log "keeping DPO: accuracy $DPO_ACC vs $BEST_ACC, loss $DPO_NLL vs $BEST_NLL"
    BEST=runs/dpo
    BEST_REPORT=runs/eval/dpo.json
    write_best "$BEST" "$BEST_REPORT" "DPO: accuracy $DPO_ACC vs $BEST_ACC, held-out loss $DPO_NLL vs $BEST_NLL"
  else
    log "discarding DPO: accuracy $DPO_ACC vs $BEST_ACC, loss $DPO_NLL vs $BEST_NLL"
  fi
fi

# 4. Summary.
python3 - "$LOOPS" "$SAMPLING" <<'PY'
import json, os, sys
loops, name = int(sys.argv[1]), sys.argv[2]
best = json.load(open("runs/best.json"))
lines = ["# Refinement summary", "",
         f"Tuned on the SFT model: r = {loops}, sampling `{name}`. Best: `{best['checkpoint']}` ({best['reason']}).", "",
         f"| Checkpoint | Held-out loss at r = {loops} | Macro accuracy | math | logic | science | JSON exact | JSON valid |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for label, path in [("SFT", "runs/eval/sft.json"), ("SFT + refine pass", "runs/eval/sft-refine.json"), ("DPO", "runs/eval/dpo.json")]:
    if not os.path.exists(path):
        lines.append(f"| {label} | not run | | | | | | |"); continue
    r = json.load(open(path))
    g = next(g for g in r["generation"] if g["loops"] == loops and g["sampling"]["name"] == name)
    c = lambda k: f"{g['configs'][k]['correct']}/{g['configs'][k]['scored']}" if k in g["configs"] else ""
    valid = g["configs"].get("structured_output", {}).get("json_valid")
    lines.append(f"| {label} | {r['nll'][f'r{loops}']:.4f} | {g['macro_accuracy']*100:.1f}% | {c('math_reasoning')} | {c('logic_reasoning')} | {c('science_knowledge')} | {c('structured_output')} | {'' if valid is None else f'{valid*100:.0f}%'} |")
sft = json.load(open("runs/eval/sft.json"))
lines += ["", "Loop dial on the SFT model (masked loss over all 2,000 eval rows, greedy macro accuracy on 50 rows per config):", "",
          "| r | loss | greedy macro |", "| ---: | ---: | ---: |"]
for key, value in sorted(sft["nll"].items(), key=lambda item: int(item[0][1:])):
    r = int(key[1:])
    acc = next((g["macro_accuracy"] for g in sft["generation"] if g["loops"] == r and g["sampling"]["name"] == "greedy"), None)
    lines.append(f"| {r} | {value:.4f} | {'' if acc is None else f'{acc*100:.1f}%'} |")
lines += ["", "Per-checkpoint details, chat smoke prompts, and sample answers: `runs/eval/*.md`."]
open("runs/eval/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
log "pipeline done; best $(python3 -c 'import json; print(json.load(open("runs/best.json"))["checkpoint"])')"
