#!/usr/bin/env bash
# Run one Boltz prediction from WSL without importing Boltz into Pharma-env.
set -euo pipefail

input_yaml="${1:?usage: run_predict.sh INPUT_YAML OUT_DIR}"
out_dir="${2:?usage: run_predict.sh INPUT_YAML OUT_DIR}"
shift 2
conda_bin="${BOLTZ_CONDA_BIN:-$HOME/miniconda3/bin/conda}"
cache_dir="${BOLTZ_CACHE:-$HOME/workspace/DiffPharma_data/boltz_cache}"

mkdir -p "$out_dir"
exec /usr/bin/time -f 'ELAPSED_SECONDS=%e' "$conda_bin" run --no-capture-output -n "${BOLTZ_ENV_NAME:-boltz-env}" \
  boltz predict "$input_yaml" --out_dir "$out_dir" --cache "$cache_dir" --use_msa_server --seed 0 "$@"
