#!/usr/bin/env bash
# Launcher for the frodo solution.
#   scripts/run.sh [-c CONFIG.toml] <view|render|camera|demo|sensors|nav|line|mission> [args...]
# Examples:
#   scripts/run.sh view --dock A
#   scripts/run.sh demo --video results/demo/pick_place.mp4
#   scripts/run.sh sensors                 # lidar plot, camera frames, odometry drift -> results/sensors/
#   scripts/run.sh mission --seed 0        # full mission -> results/nominal_seed0/
#   scripts/run.sh -c config/fast_belt.toml demo
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "-c" ]]; then
  export FRODO_CONFIG="$2"
  shift 2
fi
if [[ $# -eq 0 ]]; then
  exec uv run python -m frodo --help
fi
# Everything except the interactive viewer renders off-screen.
if [[ "$1" != "view" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi
exec uv run python -m frodo "$@"
