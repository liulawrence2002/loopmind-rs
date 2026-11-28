#!/usr/bin/env bash
# The whitepaper pipeline (docs/whitepaper.md): comparison runs at 50M tokens, the 500M
# runs, SFT, benchmarks, and the results tables in runs/whitepaper/results.md.
#
#   setsid nohup scripts/whitepaper.sh > /dev/null 2>&1 &
#   cat runs/whitepaper/STATUS; tail -f runs/whitepaper/pipeline.log
#
# Every stage leaves runs/whitepaper/<stage>.done, and a training stage resumes from its
# latest checkpoint, so running the script again continues where it stopped, including
# after a reboot. When CUDA fails (CUDA_ERROR_UNKNOWN, usually after a laptop suspend), the
# script waits and checks again every 5 minutes: reload the driver
# (sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm) or reboot and run it again.
#
# Decisions it makes, each written to runs/whitepaper/choices.env:
#   learning rate: the 50M LoopMind run at 6e-4 or 3e-4, whichever ends with the lower
#                  validation loss at r = 4.
#   loop depth:    loop mean 8 replaces mean 4 for the 500M run only if its best benchmark
#                  mean (over r) beats mean 4's by at least 0.5 points.
set -u
cd "$(dirname "$0")/.." || exit 1
BIN=${BIN:-./target/release/loopmind}
PY=${PY:-python3}
WP=runs/whitepaper
mkdir -p "$WP/configs" runs/bench runs/eval
LOG=$WP/pipeline.log
log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
status() { echo "$(date '+%F %T') $*" > "$WP/STATUS"; }
is_done() { [ -e "$WP/$1.done" ]; }
done_mark() { touch "$WP/$1.done"; }

exec 9> "$WP/lock"
if ! flock -n 9; then
  echo "another whitepaper pipeline is running" >&2
  exit 1
fi

for need in data/prepared-mix50m/pretrain/manifest.json data/prepared-500m/pretrain/manifest.json \
  data/prepared-wp/sft/manifest.json data/bench/manifest.json "$BIN"; do
  if [ ! -e "$need" ]; then
    log "missing $need; see docs/whitepaper.md, Reproducing"
    status "stopped: missing $need"
    exit 1
  fi
done

cuda_ok() {
  "$BIN" bench --checkpoint runs/pretrain --tasks piqa --limit 2 --loops 1 --device cuda \
    --out "$WP/probe.json" > "$WP/probe.log" 2>&1
}

wait_for_cuda() {
  local waited=0
  until cuda_ok; do
    if [ "$waited" = 0 ]; then
      log "CUDA is not usable ($WP/probe.log). Waiting: reload nvidia_uvm or reboot."
      waited=1
    fi
    status "waiting for CUDA: sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm, or reboot"
    sleep 300
  done
  if [ "$waited" = 1 ]; then
    log "CUDA is back"
  fi
}

trim_metrics() {
  "$PY" - "$1" << 'PY'
import json, sys
run = sys.argv[1]
step = json.load(open(f"{run}/latest.json"))["step"]
path = f"{run}/metrics.jsonl"
try:
    rows = [line for line in open(path) if json.loads(line)["step"] <= step]
except FileNotFoundError:
    rows = []
open(path, "w").writelines(rows)
PY
}

# train STAGE KIND CONFIG RUN: pretrain or sft, resuming whenever RUN/latest.json exists.
train() {
  local stage=$1 kind=$2 config=$3 run=$4
  is_done "$stage" && return 0
  local failures=0 cuda_failures=0 start
  mkdir -p "$run"
  while [ "$failures" -le 5 ] && [ "$cuda_failures" -le 20 ]; do
    wait_for_cuda
    local args=()
    if [ -e "$run/latest.json" ]; then
      trim_metrics "$run"
      args=(--resume)
    else
      rm -f "$run/metrics.jsonl"
    fi
    status "training $stage: $kind --config $config ${args[*]}"
    log "$stage: $kind --config $config ${args[*]}"
    start=$(wc -l < "$run/train.log" 2> /dev/null || echo 0)
    "$BIN" "$kind" --config "$config" "${args[@]}" >> "$run/train.log" 2>&1
    if tail -3 "$run/train.log" | grep -q "^finished"; then
      log "$stage: $(tail -1 "$run/train.log")"
      done_mark "$stage"
      return 0
    fi
    if tail -n +"$((start + 1))" "$run/train.log" | grep -q "CUDA_ERROR"; then
      cuda_failures=$((cuda_failures + 1))
      log "$stage: CUDA error"
      continue
    fi
    failures=$((failures + 1))
    log "$stage: exited without finishing ($failures)"
    sleep 60
  done
  log "$stage: giving up"
  status "stopped: $stage failed, see $run/train.log"
  exit 1
}

# bench STAGE CHECKPOINT LOOPS [bench flags]: writes runs/bench/STAGE.json and .md, and
# leaves bench-STAGE.done, so a run and its benchmarks can share a name.
bench() {
  local stage=$1 checkpoint=$2 loops=$3
  shift 3
  is_done "bench-$stage" && return 0
  local attempt
  for attempt in 1 2 3; do
    wait_for_cuda
    status "benchmarks $stage: $checkpoint at r = $loops $*"
    log "$stage: bench $checkpoint at r = $loops $*"
    if "$BIN" bench --checkpoint "$checkpoint" --loops "$loops" --device cuda \
      --out "runs/bench/$stage.json" "$@" >> "$WP/bench.log" 2>&1; then
      log "$stage: $(grep '^bench mean' "$WP/bench.log" | tail -1)"
      done_mark "bench-$stage"
      return 0
    fi
    log "$stage: bench failed (attempt $attempt, $WP/bench.log)"
    sleep 60
  done
  status "stopped: $stage failed, see $WP/bench.log"
  exit 1
}

# evaluate STAGE RUN LOOPS: the FABLE held-out eval of an SFT model, into
# runs/eval/STAGE.json. Leaves eval-STAGE.done.
evaluate() {
  local stage=$1 run=$2 loops=$3
  is_done "eval-$stage" && return 0
  local attempt
  for attempt in 1 2 3; do
    wait_for_cuda
    status "eval $stage"
    log "$stage: eval $run at r = $loops"
    if "$BIN" eval --checkpoint "$run" --out "runs/eval/$stage.json" --device cuda \
      --eval-file data/prepared-wp/sft/eval.jsonl --nll-loops "$loops" --gen-loops "$loops" \
      --sampling greedy \
      >> "$WP/eval.log" 2>&1; then
      log "$stage: $(grep '^eval recommended' "$WP/eval.log" | tail -1)"
      done_mark "eval-$stage"
      return 0
    fi
    log "$stage: eval failed (attempt $attempt, $WP/eval.log)"
    sleep 60
  done
  status "stopped: $stage failed, see $WP/eval.log"
  exit 1
}

report() {
  "$PY" scripts/whitepaper_report.py > /dev/null 2>> "$LOG" || log "report failed"
}

# last_val RUN KEY: the last validation loss at KEY (r4, r1, ...) in RUN/metrics.jsonl.
last_val() {
  "$PY" - "$1/metrics.jsonl" "$2" << 'PY'
import json, sys
value = None
for line in open(sys.argv[1]):
    row = json.loads(line)
    if row.get("val_loss"):
        value = row["val_loss"].get(sys.argv[2], value)
print(value)
PY
}

# best_mean STAGE: the best benchmark mean over loop counts in runs/bench/STAGE.json.
best_mean() {
  "$PY" -c 'import json, sys; print(max(json.load(open(sys.argv[1]))["mean"].values()))' \
    "runs/bench/$1.json"
}

# with_lr SRC DEST: SRC with the chosen peak and floor learning rates.
with_lr() {
  sed -e "s|^learning_rate = .*|learning_rate = $LR|" \
    -e "s|^min_learning_rate = .*|min_learning_rate = $MIN_LR|" "$1" > "$2"
}

DEEP="1,2,4,8,16,32"
LIMITED=(--limit 1000)

# 1. Learning rate, on LoopMind at 50M tokens.
train wp-loop4 pretrain configs/whitepaper/pretrain-50m-loop4.toml runs/wp-loop4
train wp-loop4-lr3e-4 pretrain configs/whitepaper/pretrain-50m-loop4-lr3e-4.toml runs/wp-loop4-lr3e-4
if [ ! -e "$WP/choices.env" ]; then
  high=$(last_val runs/wp-loop4 r4)
  low=$(last_val runs/wp-loop4-lr3e-4 r4)
  if "$PY" -c "import sys; sys.exit(0 if float('$high') <= float('$low') else 1)"; then
    printf 'LR=6.0e-4\nMIN_LR=6.0e-5\nLOOP4_RUN=runs/wp-loop4\n' > "$WP/choices.env"
  else
    printf 'LR=3.0e-4\nMIN_LR=3.0e-5\nLOOP4_RUN=runs/wp-loop4-lr3e-4\n' > "$WP/choices.env"
  fi
  log "learning rate: r4 val $high at 6e-4, $low at 3e-4; $(grep '^LR=' "$WP/choices.env")"
fi
# shellcheck disable=SC1091
source "$WP/choices.env"
report

# 2. The comparison runs at 50M tokens, at the chosen learning rate.
bench wp-loop4 "$LOOP4_RUN" "$DEEP"
for name in noloop unshared loop8; do
  with_lr "configs/whitepaper/pretrain-50m-$name.toml" "$WP/configs/pretrain-50m-$name.toml"
done
train wp-noloop pretrain "$WP/configs/pretrain-50m-noloop.toml" runs/wp-noloop
bench wp-noloop runs/wp-noloop 1
train wp-unshared pretrain "$WP/configs/pretrain-50m-unshared.toml" runs/wp-unshared
bench wp-unshared runs/wp-unshared 1
train wp-loop8 pretrain "$WP/configs/pretrain-50m-loop8.toml" runs/wp-loop8
bench wp-loop8 runs/wp-loop8 "$DEEP"
report

# 3. Loop depth for the 500M run.
if ! grep -q '^LOOP_MODEL=' "$WP/choices.env"; then
  four=$(best_mean wp-loop4)
  eight=$(best_mean wp-loop8)
  if "$PY" -c "import sys; sys.exit(0 if float('$eight') >= float('$four') + 0.005 else 1)"; then
    printf 'LOOP_MODEL=configs/whitepaper/model-loop8.toml\n' >> "$WP/choices.env"
  else
    printf 'LOOP_MODEL=configs/default.toml\n' >> "$WP/choices.env"
  fi
  log "loop depth: best bench mean $four at mean 4, $eight at mean 8; $(grep '^LOOP_MODEL=' "$WP/choices.env")"
fi
# shellcheck disable=SC1091
source "$WP/choices.env"

# 4. The 500M runs: LoopMind, then the no-recurrence baseline.
with_lr configs/pretrain-500m.toml "$WP/configs/pretrain-500m-loop.toml"
sed -i -e "s|^model = .*|model = \"$LOOP_MODEL\"|" -e "s|^val_loops = .*|val_loops = [1, 2, 4, 8, 16, 32]|" \
  "$WP/configs/pretrain-500m-loop.toml"
with_lr configs/whitepaper/pretrain-500m-noloop.toml "$WP/configs/pretrain-500m-noloop.toml"
train wp-500m-loop pretrain "$WP/configs/pretrain-500m-loop.toml" runs/pretrain-500m
bench wp-500m-loop runs/pretrain-500m "$DEEP"
report
train wp-500m-noloop pretrain "$WP/configs/pretrain-500m-noloop.toml" runs/wp-500m-noloop
bench wp-500m-noloop runs/wp-500m-noloop 1
report

# 5. Learning curves: every 6th checkpoint (49M tokens apart), 1,000 items per task.
for step in 1500 3000 4500 6000 7500 9000 10500 12000 13500 15000; do
  dir=$(printf 'checkpoints/step-%08d' "$step")
  bench "curve-loop-$step" "runs/pretrain-500m/$dir" 1,4,8 "${LIMITED[@]}"
  bench "curve-noloop-$step" "runs/wp-500m-noloop/$dir" 1 "${LIMITED[@]}"
done
report

# 6. SFT both 500M models on FABLE + smol-smoltalk, then the held-out eval and benchmarks.
train wp-sft-loop sft configs/whitepaper/sft-500m-loop.toml runs/wp-sft-loop
evaluate wp-sft-loop runs/wp-sft-loop 1,2,4,8
bench wp-sft-loop runs/wp-sft-loop "$DEEP"
train wp-sft-noloop sft configs/whitepaper/sft-500m-noloop.toml runs/wp-sft-noloop
evaluate wp-sft-noloop runs/wp-sft-noloop 1
bench wp-sft-noloop runs/wp-sft-noloop 1
report

# 7. A second seed for the main 50M pair, to measure run-to-run spread.
with_lr configs/whitepaper/pretrain-50m-loop4.toml "$WP/configs/pretrain-50m-loop4-s7.toml"
with_lr configs/whitepaper/pretrain-50m-noloop.toml "$WP/configs/pretrain-50m-noloop-s7.toml"
sed -i -e 's|^seed = .*|seed = 7|' -e 's|^run_dir = .*|run_dir = "runs/wp-loop4-s7"|' "$WP/configs/pretrain-50m-loop4-s7.toml"
sed -i -e 's|^seed = .*|seed = 7|' -e 's|^run_dir = .*|run_dir = "runs/wp-noloop-s7"|' "$WP/configs/pretrain-50m-noloop-s7.toml"
train wp-loop4-s7 pretrain "$WP/configs/pretrain-50m-loop4-s7.toml" runs/wp-loop4-s7
bench wp-loop4-s7 runs/wp-loop4-s7 "$DEEP"
train wp-noloop-s7 pretrain "$WP/configs/pretrain-50m-noloop-s7.toml" runs/wp-noloop-s7
bench wp-noloop-s7 runs/wp-noloop-s7 1
report

log "all stages done; results in $WP/results.md"
status "done: results in $WP/results.md"
