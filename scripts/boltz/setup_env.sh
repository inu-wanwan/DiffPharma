#!/usr/bin/env bash
# Create the conda environment that runs Boltz-2 as DiffPharma's affinity oracle.
#
# Boltz-2 CANNOT be installed into Pharma-env. Every dependency axis conflicts,
# and two of them are exact pins:
#
#            Pharma-env          boltz 2.2.1
#   torch    2.1.1 (cu118)       >=2.2
#   numpy    1.22.3              >=1.26,<2.0
#   lightning 1.7.4              ==2.5.0   (exact)
#   rdkit    2022.03.2           >=2024.3.2
#   scipy    1.7.3               ==1.13.1  (exact)
#
# Upgrading Pharma-env is not an option either: the distributed
# DiffPharma_best.ckpt is a pytorch-lightning 1.7.4 checkpoint. So the two
# stacks talk over a process boundary, never an import. This script builds the
# far side of that boundary; analysis/boltz_affinity.py is the near side.
#
# Usage:
#   BOLTZ_SRC=/path/to/boltz/clone bash scripts/boltz/setup_env.sh
set -euo pipefail

BOLTZ_SRC="${BOLTZ_SRC:?set BOLTZ_SRC to the pinned boltz clone (e.g. ../boltz)}"
ENV_NAME="${BOLTZ_ENV_NAME:-boltz-env}"
PY_VERSION="${BOLTZ_PY_VERSION:-3.11}"
# Weights, CCD and MSA caches are several GB and are read on every call. Keep
# them on ext4: /mnt/c is slow enough to dominate a 20 s/ligand oracle.
CACHE_DIR="${BOLTZ_CACHE:-$HOME/workspace/DiffPharma_data/boltz_cache}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

test -f "$BOLTZ_SRC/pyproject.toml" || {
    echo "not a boltz checkout: $BOLTZ_SRC" >&2
    exit 1
}
# shellcheck disable=SC1090
source "$CONDA_SH"

commit=$(git -C "$BOLTZ_SRC" rev-parse HEAD)
describe=$(git -C "$BOLTZ_SRC" describe --tags --always)
dirty=$(git -C "$BOLTZ_SRC" status --porcelain | wc -l)
echo "boltz source : $BOLTZ_SRC"
echo "boltz version: $describe ($commit, dirty=$dirty)"
echo "conda env    : $ENV_NAME (python $PY_VERSION)"
echo "cache dir    : $CACHE_DIR"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "env $ENV_NAME already exists; reusing it"
else
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION" pip
fi

# Installed non-editable on purpose: the clone stays a pristine reference and
# the import path lives on ext4. Provenance comes from the recorded commit,
# not from the install mode.
conda run --no-capture-output -n "$ENV_NAME" python -m pip install --upgrade pip
# The base package deliberately omits NVIDIA's cuEquivariance kernels.  They
# are required unless every prediction uses --no_kernels, so install Boltz's
# CUDA extra explicitly for the RTX 4090 / H100 execution path.
conda run --no-capture-output -n "$ENV_NAME" python -m pip install "${BOLTZ_SRC}[cuda]"

mkdir -p "$CACHE_DIR"

# Record exactly what was built, for the Phase 0 "fix and record the measurement
# conditions" invariant.
prefix=$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')
{
    echo "boltz_source=$BOLTZ_SRC"
    echo "boltz_commit=$commit"
    echo "boltz_describe=$describe"
    echo "boltz_dirty=$dirty"
    echo "conda_env=$ENV_NAME"
    echo "conda_prefix=$prefix"
    echo "python=$(conda run -n "$ENV_NAME" python -V 2>&1)"
    echo "torch=$(conda run -n "$ENV_NAME" python -c 'import torch; print(torch.__version__)')"
    echo "cuda_available=$(conda run -n "$ENV_NAME" python -c 'import torch; print(torch.cuda.is_available())')"
    echo "built_at=$(date -Iseconds)"
} > "$CACHE_DIR/provenance.txt"

echo
echo "--- provenance ---"
cat "$CACHE_DIR/provenance.txt"
echo
echo "boltz CLI: $(conda run -n "$ENV_NAME" which boltz)"
