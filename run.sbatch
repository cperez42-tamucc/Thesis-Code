#!/usr/bin/env bash
#SBATCH --job-name=gaia-svgpr-raw
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodelist=crest-g002
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --time=23:59:00
#SBATCH --array=0-5%1
#SBATCH --output=slurm-gaia-raw-%A_%a.out
#SBATCH --error=slurm-gaia-raw-%A_%a.err

# Experiment array: 0 mock-hapitas; 1 mock-neural; 2 subset-hapitas;
# 3 subset-neural; 4 full-neural; 5 full-spacefill.
# 0-5%1 serializes the tasks on a single H100 with up to 24h each.
# First run prepare_raw_gaia.sbatch to convert the actual VOTable on CPU.
# This script will also prepare the cache if missing, but GPU time is wasted.
set -euo pipefail

RUN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$RUN_DIR"
MODES=(mock-hapitas mock-neural subset-hapitas subset-neural full-neural full-spacefill)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$TASK_ID" =~ ^[0-5]$ ]] || { echo "ERROR: array index must be 0-5" >&2; exit 2; }
MODE="${MODE:-${MODES[$TASK_ID]}}"

SCRIPT="${BENCHMARK_SCRIPT:-$RUN_DIR/gaia_six_experiment_raw_votable.py}"
GAIA_VOT="${GAIA_VOT:-/home/cperez85/neural tests/2cb451da-d64b-11f1-b71e-bc97e148b76b-O-result.vot}"
RAW_CACHE="${RAW_CACHE:-/home/cperez85/neural tests/gaia_from_raw_votable.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/six_experiment_outputs_raw_votable}"
mkdir -p "$OUTPUT_DIR"
[[ -f "$SCRIPT" ]] || { echo "ERROR: missing script $SCRIPT" >&2; exit 3; }
[[ -f "$RUN_DIR/gaia_raw_votable.py" ]] || { echo "ERROR: missing gaia_raw_votable.py" >&2; exit 3; }
if [[ "$MODE" != mock-* && ! -f "$GAIA_VOT" ]]; then
  echo "ERROR: raw Gaia archive download missing: $GAIA_VOT" >&2
  echo "The TAP query job name does not automatically download its result to Crest." >&2
  exit 4
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON="$CONDA_PREFIX/bin/python"
else
  PYTHON="$HOME/.conda/envs/svgpr/bin/python"
fi
[[ -x "$PYTHON" ]] || { echo "ERROR: missing Python interpreter $PYTHON" >&2; exit 5; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export CUDA_DEVICE_ORDER=PCI_BUS_ID

EPOCHS="${EPOCHS:-300}"
INDUCING="${INDUCING:-1000}"
BATCH_CAP="${BATCH_CAP:-8000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
echo "=== Gaia raw VOTable SVGPR task ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} array index=$TASK_ID mode=$MODE"
echo "node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-auto}"
echo "interpreter=$PYTHON"
echo "raw_gaia=$GAIA_VOT"
echo "cache=$RAW_CACHE"
echo "results=$OUTPUT_DIR"
echo "epochs=$EPOCHS inducing=$INDUCING batch_cap=$BATCH_CAP"
date -Is
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi
"$PYTHON" - <<'PY'
import torch, gpytorch, svgpr, astropy
assert torch.cuda.is_available(), 'No CUDA GPU detected'
print('Torch:',torch.__version__,'GPyTorch:',gpytorch.__version__,'Astropy:',astropy.__version__)
print('GPU:',torch.cuda.get_device_name(0))
print('svgpr:',svgpr.__file__)
PY
ARGS=(--mode "$MODE" --gaia "$GAIA_VOT" --raw-cache "$RAW_CACHE"
      --output "$OUTPUT_DIR" --seed 25 --inducing "$INDUCING"
      --epochs "$EPOCHS" --batch-ratio 100 --batch-cap "$BATCH_CAP"
      --mean-lr 1.0 --disp-lr 0.1 --variational natural
      --checkpoint-every "$CHECKPOINT_EVERY" --dtype float64 --resume)
[[ "${NO_REFIT_FULL:-0}" == 1 ]] && ARGS+=(--no-refit-full)
echo "Starting $MODE"
"$PYTHON" -u "$SCRIPT" "${ARGS[@]}"
echo "Finished $MODE"
date -Is
