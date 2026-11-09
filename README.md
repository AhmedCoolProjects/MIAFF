# MIAFF: Multimodal Infrastructure-Aware Fusion Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official research repository for **MIAFF** (Multimodal Infrastructure-Aware Fusion Framework), a framework that synergistically fuses lexical domain representations (character-level NNs and domain-adapted PLMs) with infrastructure metadata (DNS, RDAP/WHOIS, TLS, IP/Hosting, and Cross-signal correlations) for robust malicious domain detection and family attribution.

---

## Repository Structure

```text
miaff/
├── 01_metadata_selection/     # Data loaders, engineered metadata features, split handlers
├── 02_NN_Models/              # Character-level neural architectures (CharCNN, CharBiGRU, etc.)
├── 03_LANG_Models/            # Pre-trained language model wrappers (DomURLs_BERT, SecureBERT, SecBERT)
├── 04_Learning/               # Core training and evaluation engines (Baselines & MIAFF Fusion)
├── 05_Reporting/              # Benchmark analysis, metrics computation, and LaTeX table generators
├── 06_Interpreting/           # Model interpretability (Integrated Gradients and SHAP attribution)
├── 10_runs/                   # Reproducibility SLURM / shell execution runners
├── paper_checkpoints/        # Pretrained PyTorch models (CharCNN, CharBiGRU, Metadata_MLP across 9 datasets)
├── requirements.txt           # Python package dependencies
└── .env.example               # Template environment configuration (dataset paths, W&B credentials)
```

---

## Quickstart & Environment Setup

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/AhmedCoolProjects/MIAFF.git
cd MIAFF
pip install -r requirements.txt
```

### 2. Configuration
Copy the `.env.example` template:
```bash
cp .env.example .env
```
Configure your dataset paths and optional `WANDB_API_KEY` inside `.env` or adapt `config.yml`.

---

## Reproducing Paper Experiments

### 1. Training Baseline Models
- **Lexical Character Baselines (`CharCNN`, `CharBiGRU`):**
  ```bash
  python 04_Learning/BASELINE_Char.py --config config.yml --dataset threatfox_malware_binary
  ```
- **Pretrained Language Models (`DomURLs_BERT`, `SecureBERT`):**
  ```bash
  python 04_Learning/BASELINE_LANG.py --config config.yml --dataset threatfox_malware_binary
  ```
- **Infrastructure Metadata-Only MLP:**
  ```bash
  python 04_Learning/Metadata_MLP.py --config config.yml --dataset threatfox_malware_binary
  ```

### 2. Training MIAFF Fusion
Train lexical models fused with infrastructure metadata via MIAFF:
```bash
# Character-level fusion
python 04_Learning/Fusion.py --config config.yml --dataset threatfox_malware_binary --combination N_cat_N_ce

# Pretrained Language Model fusion
python 04_Learning/Fusion.py --config config.yml --dataset threatfox_malware_binary --encoder lang --combination N_cat_N_ce
```

### 3. Interpretability & Feature Attribution
Compute true-class Integrated Gradients (IG) attributions across string representations and metadata categories:
```bash
python 06_Interpreting/IG_categories_fusion_all.py
python 06_Interpreting/IG_features_fusion_all.py
```
---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
