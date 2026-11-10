#!/usr/bin/env bash
# Launcher for the ringcell solution.
#   scripts/run.sh [-c CONFIG.toml] <view|render|camera|demo|mission> [args...]
# Examples:
#   scripts/run.sh view --dock A
#   scripts/run.sh demo --video results/demo/pick_place.mp4
#   scripts/run.sh -c config/fast_belt.toml demo
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "-c" ]]; then
  export RINGCELL_CONFIG="$2"
  shift 2
fi
if [[ $# -eq 0 ]]; then
  exec uv run python -m ringcell --help
fi
# Everything except the interactive viewer renders off-screen.
if [[ "$1" != "view" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi
exec uv run python -m ringcell "$@"
