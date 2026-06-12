# ToME (R=0.9) — how we reproduced it

Same idea as EViT, just **token merging instead of pruning: plain CLIP + ToME, no TCA reservoir.**

- **Script:** `scripts/run_tome.sh`
- **Command (per dataset):**
  `python3 clip_zeroshot.py --backbone ViT-B/16 --token_pruning ToME-0.1 --datasets <ds> --data-root TCA/data`
- **Key setting:** `ToME-0.1` = merge a 10% fraction of tokens at layers 3/6/9 (the paper's R=0.9). No reservoir, no logits correction — `clip_zeroshot.py` does `CLIP forward → cosine sim → top-1`.
- Same prompts/splits/preprocessing as our CLIP run; batch size 1. Deterministic.
- **Output:** `results/ToME_R0.9_<ds>.txt` (accuracy + GFLOPs).

Paper reference: Table 1 row **ToME_{R=0.9}** (avg 64.88, GFLOPs 15.31).
