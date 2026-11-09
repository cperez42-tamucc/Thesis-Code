#!/usr/bin/env bash
#SBATCH --job-name=gaia-vot-prep
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-gaia-vot-prep-%j.out
#SBATCH --error=slurm-gaia-vot-prep-%j.err

# CPU-only, one-time raw Gaia VOTable preprocessing; avoids wasting H100 time.
set -euo pipefail
RUN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$RUN_DIR"
SCRIPT="${BENCHMARK_SCRIPT:-$RUN_DIR/gaia_six_experiment_raw_votable.py}"
GAIA_VOT="${GAIA_VOT:-/home/cperez85/neural tests/2cb451da-d64b-11f1-b71e-bc97e148b76b-O-result.vot}"
RAW_CACHE="${RAW_CACHE:-/home/cperez85/neural tests/gaia_from_raw_votable.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/six_experiment_outputs_raw_votable}"
[[ -f "$GAIA_VOT" ]] || { echo "ERROR: raw Gaia archive file missing: $GAIA_VOT" >&2; exit 2; }
[[ -f "$RUN_DIR/gaia_raw_votable.py" ]] || { echo 'Missing gaia_raw_votable.py' >&2; exit 3; }
if [[ -n "${PYTHON_BIN:-}" ]]; then
 PYTHON="$PYTHON_BIN"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
 PYTHON="$CONDA_PREFIX/bin/python"
else
 PYTHON="$HOME/.conda/envs/svgpr/bin/python"
fi
mkdir -p "$OUTPUT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
echo "Preparing raw Gaia on $(hostname) at $(date -Is)"
echo "Source: $GAIA_VOT"
echo "Cache:  $RAW_CACHE"
"$PYTHON" -u "$SCRIPT" --gaia "$GAIA_VOT" --raw-cache "$RAW_CACHE" \
  --output "$OUTPUT_DIR" --prepare-gaia-only --raw-chunk-rows 65536
