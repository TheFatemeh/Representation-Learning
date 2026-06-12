# EViT (R=0.9) — how we reproduced it

EViT is a **pure token-pruning baseline: plain CLIP + EViT pruning, no TCA reservoir/adaptation.**
So we just ran CLIP zero-shot with the visual encoder loaded in EViT mode.

- **Script:** `scripts/run_evit.sh`
- **Command (per dataset):**
  `python3 clip_zeroshot.py --backbone ViT-B/16 --token_pruning EViT-0.1 --datasets <ds> --data-root TCA/data`
- **Key setting:** `EViT-0.1` = drop 10% of tokens at layers 3/6/9 (this is the paper's R=0.9). No reservoir, no logits correction — `clip_zeroshot.py` just does `CLIP forward → cosine sim → top-1`.
- Same prompts/splits/preprocessing as our CLIP run; batch size 1. Deterministic (no online cache).
- **Output:** `results/EViT_R0.9_<ds>.txt` (accuracy + GFLOPs).

Paper reference: Table 1 row **EViT_{R=0.9}** (avg 65.17, GFLOPs 15.41).
