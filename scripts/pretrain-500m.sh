#!/usr/bin/env bash
# Pretrain on the 500M-token mix (configs/pretrain-500m.toml) and resume after a crash.
#
#   cargo build --release -p loopmind --features cuda
#   ./target/release/loopmind prepare-pretrain --config configs/data-500m.toml
#   setsid nohup scripts/pretrain-500m.sh > /dev/null 2>&1 &
#   tail -f runs/pretrain-500m/train.log
#
# Running it again continues the run. When runs/pretrain-500m/latest.json exists, every
# attempt is a --resume, including the first. metrics.jsonl is trimmed back to the
# checkpoint first, so the curve has no repeated steps. After 5 failed retries it gives up.
#
# To pause: kill this script first, then the trainer. Run the script again to continue.
# After a laptop suspend, CUDA can fail with CUDA_ERROR_UNKNOWN until the driver is
# reloaded (sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm) or the machine reboots.
set -u
cd "$(dirname "$0")/.." || exit 1
BIN=${BIN:-./target/release/loopmind}
CONFIG=configs/pretrain-500m.toml
RUN=runs/pretrain-500m
DATA=data/prepared-500m/pretrain
mkdir -p "$RUN"
LOG=$RUN/train.log
note() { echo "=== $* $(date '+%F %T') ===" | tee -a "$LOG"; }

if [ ! -e "$DATA/manifest.json" ]; then
  echo "no $DATA/manifest.json: run prepare-pretrain --config configs/data-500m.toml first" >&2
  exit 1
fi
if pgrep -f "^$BIN pretrain --config $CONFIG" > /dev/null; then
  echo "a trainer for $CONFIG is already running" >&2
  exit 1
fi

trim_metrics() {
  python3 - "$RUN" << 'PY'
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

for attempt in 0 1 2 3 4 5; do
  args=()
  if [ -e "$RUN/latest.json" ]; then
    trim_metrics
    args=(--resume)
  else
    rm -f "$RUN/metrics.jsonl"
  fi
  note "attempt $attempt ${args[*]}"
  start=$(wc -l < "$LOG")
  "$BIN" pretrain --config "$CONFIG" "${args[@]}" >> "$LOG" 2>&1
  if tail -3 "$LOG" | grep -q "^finished"; then
    note "finished"
    exit 0
  fi
  if tail -n +"$((start + 1))" "$LOG" | grep -q "CUDA_ERROR_UNKNOWN"; then
    note "CUDA_ERROR_UNKNOWN, usually after a suspend: reload nvidia_uvm or reboot, then run this again"
    exit 1
  fi
  note "exited without finishing"
  sleep 120
done
note "giving up"
exit 1
