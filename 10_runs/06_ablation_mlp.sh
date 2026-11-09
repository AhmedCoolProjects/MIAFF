#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="domurls"
if command -v conda >/dev/null 2>&1; then
  conda activate "" 2>/dev/null || true
fi
cd "$PROJECT_ROOT"

python 06_Interpreting/Ablation.py \
  --config config.yml \
  --mode drop