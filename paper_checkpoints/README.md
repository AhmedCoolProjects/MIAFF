# Paper Checkpoints

This directory contains the trained neural network model checkpoints used in the paper evaluation across all 9 benchmark datasets.

## Included Models (45 Native PyTorch Checkpoints)

Each file is stored in native PyTorch format (`best.pt`, $< 4$ MB per checkpoint):

```text
paper_checkpoints/
├── Metadata_MLP/               # Infrastructure Metadata-Only MLPs (9 datasets)
│   ├── dibhran_malware_binary/best.pt
│   ├── threatfox_malware_binary/best.pt
│   ├── duben_malware_binary/best.pt
│   ├── macformer_malware_binary/best.pt
│   ├── dibhran_phishing_binary/best.pt
│   ├── duben_phishing_binary/best.pt
│   ├── macformer_phishing_binary/best.pt
│   ├── dibhran_malware_family/best.pt
│   └── threatfox_malware_family/best.pt
├── CharCNN/
│   ├── baseline/               # String-only CharCNN baselines (9 datasets)
│   └── miaff_fusion/           # MIAFF (N_cat_N_ce) multimodal fusion (9 datasets)
└── CharBiGRU/
    ├── baseline/               # String-only CharBiGRU baselines (9 datasets)
    └── miaff_fusion/           # MIAFF (N_cat_N_ce) multimodal fusion (9 datasets)
```

## How to Load a Checkpoint

```python
import torch

# Load checkpoint dictionary
ckpt_path = "paper_checkpoints/CharCNN/miaff_fusion/threatfox_malware_binary/best.pt"
checkpoint = torch.load(ckpt_path, map_location="cpu")

# Inspect keys
print("Keys in checkpoint:", list(checkpoint.keys()))
# Typical keys: ['epoch', 'model_state_dict', 'optimizer_state_dict', 'vocab_size', 'model_info', 'val_metrics']
```

## Pre-trained Language Models (PLMs)

Fine-tuned Pre-trained Language Models (`DomURLs_BERT`, `SecureBERT`, `CySecBERT`) are approximately 1.4 GB per checkpoint, which exceeds GitHub's 100 MB single-file limit. 

To reproduce or fine-tune PLM baselines and MIAFF PLM fusion:
```bash
# PLM String Baseline
python 04_Learning/BASELINE_LANG.py --config config.yml --dataset threatfox_malware_binary

# PLM MIAFF Fusion
python 04_Learning/Fusion.py --config config.yml --dataset threatfox_malware_binary --encoder lang --combination N_cat_N_ce
```
