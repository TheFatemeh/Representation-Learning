# TCA (R=0.9) — how we reproduced it

We simply ran the **original TCA repo's `runner.py`** — no changes to the method.

- **Script:** `scripts/run_tca_repro.sh`
- **Command (per dataset):**
  `python3 runner.py --backbone ViT-B/16 --token_pruning Ours-0.035 --datasets <ds> --data-root TCA/data`
- **Key setting:** `Ours-0.035` *is* the paper's **R=0.9** setting (FLOP-matched to EViT-0.1). Token reduction + the domain-aware reservoir + logits correction are all on (the repo defaults).
- Per-dataset hyperparameters (`reservoir_size`, `scale`, `lambd`, `beta`) come from the repo's `configs/<ds>.yaml`. Batch size 1, no augmentation, seed 1.
- **Output:** `results/TCA_R0.9_<ds>.txt` (accuracy + GFLOPs).

Paper reference: Table 1 row **TCA_{R=0.9}** (avg 68.69, GFLOPs 15.45).
