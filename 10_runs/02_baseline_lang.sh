#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="domurls"
if command -v conda >/dev/null 2>&1; then
  conda activate "" 2>/dev/null || true
fi
cd "$PROJECT_ROOT"

datasets=(
  threatfox_malware_binary
  # macformer_malware_binary
  # duben_malware_binary
  # dibhran_malware_binary
  # macformer_phishing_binary
  # duben_phishing_binary
  dibhran_phishing_binary
  threatfox_malware_family
  dibhran_malware_family
)

for dataset in "${datasets[@]}"; do
  python 04_Learning/BASELINE_LANG.py \
    --config config.yml \
    --dataset "$dataset"
done