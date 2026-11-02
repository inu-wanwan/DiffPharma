#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
conda_bin="${BOLTZ_CONDA_BIN:-$HOME/miniconda3/bin/conda}"
benchmark_dir="${BOLTZ_PHASE0_DIR:-$HOME/workspace/DiffPharma_runs/boltz_phase0_benchmark}"
cache_dir="${BOLTZ_CACHE:-$HOME/workspace/DiffPharma_data/boltz_cache}"

exec "$conda_bin" run --no-capture-output -n "${BOLTZ_ENV_NAME:-boltz-env}" \
  python "$repo_dir/scripts/boltz/run_phase0_benchmark.py" \
  --benchmark-dir "$benchmark_dir" --cache "$cache_dir" --no-kernels
