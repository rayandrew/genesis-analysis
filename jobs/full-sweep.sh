#!/bin/bash -e
# Count the runs of the full genesis sweep and print how amg's main scales on matrix.

#FLUX: --job-name=full-sweep
#FLUX: -N 1
#FLUX: -t 30m
#FLUX: -q pdebug
#FLUX: --output=full-sweep-{{id}}.out

# ---- 1. Settings: change these ----
TRACE=/usr/workspace/genesis-wisdom/dftracer-traces/genesis-sweep/genesis.pfw.gz
INDEX="$PWD/data/full-index"
AMG_INPUT=p1_n64x64x32
export DFTRACER_UTILS_LOG_LEVEL=warn

# ---- 2. Record the run ----
echo "Job $(flux getattr jobid) on $(hostname) at $(date)"

# ---- 3. Run ----
mkdir -p "$INDEX"
# The environment from uv sync; the job does not rebuild it.
uv run --no-sync python scripts/full_sweep.py "$TRACE" "$INDEX" "$AMG_INPUT"
