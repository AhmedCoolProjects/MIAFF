#!/usr/bin/env python3
"""SHAP interpretability across modality, category, and feature levels."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from IG import forward_fusion_from_embeds, run_interpretability

METHOD_ABBR = "SHAP"
BG_SIZE = 128
NSAMPLES = 32
SHAP_BATCH_SIZE = 256
# ponytail: vectorized Expected Gradients (GradientExplainer) with 128 train background samples and 32 MC draws; increase NSAMPLES if variance is high.


class FusionEmbedWrapper(nn.Module):
    def __init__(self, fusion_model):
        super().__init__()
        self.fusion_model = fusion_model

    def forward(self, char_embed, metadata):
        return forward_fusion_from_embeds(self.fusion_model, char_embed, metadata)


def _ensure_bg(model, train_ds, device, is_fusion=False):
    if not hasattr(model, "_shap_bg"):
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(len(train_ds), size=min(BG_SIZE, len(train_ds)), replace=False)
        with torch.no_grad():
            bg_meta = train_ds.metadata[bg_idx].to(device)
            if is_fusion:
                bg_ids = train_ds.inputs[bg_idx].to(device)
                bg_embed = model.text_encoder.embedding(bg_ids)
                model._shap_bg = (bg_embed, bg_meta)
            else:
                model._shap_bg = bg_meta
    return model._shap_bg


def shap_mlp_batch(mlp_model, train_ds, meta_batch, target_classes, device, nsamples=NSAMPLES):
    bg_meta = _ensure_bg(mlp_model, train_ds, device, is_fusion=False)
    bsz, feat_dim = meta_batch.shape
    num_bg = bg_meta.size(0)
    gen = torch.Generator(device=device).manual_seed(42)

    grad_sum = torch.zeros_like(meta_batch)
    diff_sum = torch.zeros_like(meta_batch)
    ar = torch.arange(bsz, device=device)

    for _ in range(nsamples):
        ridx = torch.randint(0, num_bg, (bsz,), generator=gen, device=device)
        ref = bg_meta[ridx]
        alpha = torch.rand((bsz, 1), generator=gen, device=device)
        diff = meta_batch - ref
        interp = (ref + alpha * diff).detach().requires_grad_(True)
        logits = mlp_model(interp)[ar, target_classes]
        g = torch.autograd.grad(logits.sum(), interp)[0].detach()
        grad_sum += g * diff
    return grad_sum / nsamples


def shap_fusion_batch(
    fusion_model, train_ds, char_ids_batch, meta_batch, target_classes, device, nsamples=NSAMPLES
):
    bg_embed, bg_meta = _ensure_bg(fusion_model, train_ds, device, is_fusion=True)
    bsz = char_ids_batch.size(0)
    num_bg = bg_meta.size(0)
    with torch.no_grad():
        sample_embed = fusion_model.text_encoder.embedding(char_ids_batch)

    gen = torch.Generator(device=device).manual_seed(42)
    attr_e = torch.zeros_like(sample_embed)
    attr_m = torch.zeros_like(meta_batch)
    ar = torch.arange(bsz, device=device)

    for _ in range(nsamples):
        ridx = torch.randint(0, num_bg, (bsz,), generator=gen, device=device)
        ref_e = bg_embed[ridx]
        ref_m = bg_meta[ridx]
        alpha = torch.rand((bsz, 1), generator=gen, device=device)
        diff_e = sample_embed - ref_e
        diff_m = meta_batch - ref_m

        step_e = (ref_e + alpha.unsqueeze(-1) * diff_e).detach().requires_grad_(True)
        step_m = (ref_m + alpha * diff_m).detach().requires_grad_(True)
        logits = forward_fusion_from_embeds(fusion_model, step_e, step_m)[ar, target_classes]
        ge, gm = torch.autograd.grad(logits.sum(), (step_e, step_m))
        attr_e += ge.detach() * diff_e
        attr_m += gm.detach() * diff_m

    return attr_e / nsamples, attr_m / nsamples


def main():
    run_interpretability(
        METHOD_ABBR,
        shap_fusion_batch,
        shap_mlp_batch,
        need_train_bg=True,
        batch_size=SHAP_BATCH_SIZE,
    )


if __name__ == "__main__":
    main()
