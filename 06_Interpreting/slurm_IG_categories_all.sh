#!/bin/bash
#SBATCH --job-name=miaff_ig_categories
#SBATCH --account=your_account
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=09_output/06_Interpreting/ig_categories_%j.out
#SBATCH --error=09_output/06_Interpreting/ig_categories_%j.err

set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
python3 06_Interpreting/IG_categories_all.py --execute
