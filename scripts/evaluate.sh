#!/usr/bin/env bash
# Runs the five published acceptance runs and checks each one.
#   scripts/evaluate.sh            # all five, in parallel (about 10 minutes on 8 cores)
#   scripts/evaluate.sh human-stop # just one of: nominal-seed-0/1/2, human-stop, plc-retry
# Output per run in evidence/<run>/: events.jsonl (evidence log), summary.json,
# evaluation.json (customer-side checks), seat_samples.jsonl.gz (raw seat measurements),
# run.log, check.json (starter systems-check); plc-retry adds the socket trace and the
# controller ledger; human-stop adds demo.mp4.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"

declare -A ARGS=(
  [nominal-seed-0]="--scenario nominal --seed 0"
  [nominal-seed-1]="--scenario nominal --seed 1"
  [nominal-seed-2]="--scenario nominal --seed 2"
  [human-stop]="--scenario human_stop --seed 0 --video evidence/human-stop/demo.mp4"
  [plc-retry]="--scenario plc_retry --seed 0"
)
declare -A CHECK=([human-stop]=human_stop [plc-retry]=plc_retry)
RUNS=("${@:-nominal-seed-0 nominal-seed-1 nominal-seed-2 human-stop plc-retry}")
read -r -a RUNS <<< "${RUNS[*]}"

run_one() {
  local name=$1 out=evidence/$1
  rm -rf "$out" && mkdir -p "$out"
  # shellcheck disable=SC2086
  uv run python -m frodo mission ${ARGS[$name]} --port 0 --out "$out" > "$out/run.log" 2>&1
  uv run systems-check "$out/events.jsonl" --scenario "${CHECK[$name]:-nominal}" \
    > "$out/check.json" || true
  gzip -f "$out/seat_samples.jsonl"
}

for name in "${RUNS[@]}"; do run_one "$name" & done
wait
for name in "${RUNS[@]}"; do
  uv run python - "$name" <<'PY'
import json, sys
name = sys.argv[1]
s = json.load(open(f"evidence/{name}/summary.json"))
c = json.load(open(f"evidence/{name}/check.json"))
counts = {k: v["accepted"] for k, v in s["counts"].items()}
print(f"{name:15s} accepted {counts}  {s['simulated_s']:6.1f} s sim  {s['wall_s']:5.0f} s wall  "
      f"systems-check {'PASS' if c['passed'] else 'FAIL'}  "
      f"evaluator {'PASS' if s['evaluator']['passed'] else 'FAIL ' + str(s['evaluator']['failed_checks'])}")
PY
done
