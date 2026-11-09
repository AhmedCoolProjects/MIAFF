#!/bin/bash
#SBATCH --job-name=miaff_infer_bench
#SBATCH --account=your_account
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=09_output/05_Reporting/slurm_infer_bench_%j.out
#SBATCH --error=09_output/05_Reporting/slurm_infer_bench_%j.err
#SBATCH --exclusive

# Dedicated inference benchmark: baseline vs MIAFF_base (N_cat_N_ce).
# Single A100 GPU for the whole run -> identical hardware for every pair.
# 10 warmup + 10 measured repetitions, fixed per-family batch sizes.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "node: $(hostname)"
echo "start: $(date -u +%FT%TZ)"
python3 -u 05_Reporting/inference_benchmark.py \
  --repeats 10 --warmup 10 --device cuda
echo "end: $(date -u +%FT%TZ)"
