"""Standalone GFLOPs measurement for the CLIP visual encoder under a given token-pruning setting.

GFLOPs are dataset/content-independent (the prune/merge schedule is fixed by backbone + drop-rate),
so ONE measurement gives the value reported in Table 1 for that method.
Requires a GPU (the repo's attention blocks hardcode .cuda() in the forward pass).

Examples:
    python measure_gflops.py --token_pruning Ours-0.0     # == CLIP baseline (R=1.0)  -> ~17.59
    python measure_gflops.py --token_pruning Ours-0.035   # == TCA R=0.9              -> ~15.45
    python measure_gflops.py --token_pruning EViT-0.1      # == EViT R=0.9
"""
import argparse
import logging

import torch
import clip
from fvcore.nn import FlopCountAnalysis


def measure_visual_gflops(clip_model):
    logging.getLogger("fvcore.nn.jit_analysis").setLevel(logging.ERROR)
    w = clip_model.visual.conv1.weight
    res = clip_model.visual.input_resolution
    dummy = torch.zeros(1, 3, res, res, dtype=w.dtype, device=w.device)
    cache_bak = clip_model.visual.cls_token_cache
    clip_model.visual.cls_token_cache = None
    with torch.no_grad():
        fca = FlopCountAnalysis(clip_model.visual, dummy)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        total = fca.total() / 1e9
    clip_model.visual.cls_token_cache = cache_bak
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', default='ViT-B/16')
    p.add_argument('--token_pruning', default='Ours-0.035',
                   help='e.g. Ours-0.0 (=CLIP/R=1.0), Ours-0.035 (=TCA R=0.9), EViT-0.1')
    args = p.parse_args()
    model, _ = clip.load(args.backbone, args.token_pruning)  # default device = cuda
    model.eval()
    g = measure_visual_gflops(model)
    print(f"{args.backbone}  {args.token_pruning}:  GFLOPs (visual encoder, per image) = {g:.2f}")


if __name__ == '__main__':
    main()
