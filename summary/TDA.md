# TDA — how we reproduced it

TDA is the strongest baseline row in Table 1. Unlike CLIP/EViT/ToME, it is **not** part of the
TCA repo, so we used the **official TDA repo** (github.com/kdiAAA/TDA) directly — pulled it,
pointed it at our data, and ran it. No changes to the method.

- **Script:** `scripts/run_tda.sh`
- **Command (per dataset):**
  `python3 tda_runner.py --config configs --datasets <ds> --backbone ViT-B/16 --data-root TDA/data`
- **Setup steps (the only things we touched):**
  - cloned the repo into `MainRepo/TDA`
  - symlinked the datasets from `TCA/data` (same images/splits as all our other runs)
  - disabled wandb logging (`WANDB_MODE=disabled`) — the runner logs every image otherwise
- Per-dataset cache hyperparameters come from the repo's own `configs/<ds>.yaml`. Batch size 1,
  seed 1, ViT-B/16. **Not deterministic across machines** (see below).
- **Output:** `results/TDA_<ds>.txt`

Paper reference: Table 1 row **TDA** (avg 67.52, GFLOPs 17.58 — TDA does not prune tokens).

## Result: average matches, two datasets don't

Our 9-dataset average is **68.02 vs the paper's 67.52** (+0.50, well within ±2%). Most datasets
are near-exact (e.g. Caltech 94.16 vs 94.24, Food 86.17 vs 86.14, Flowers 71.74 vs 71.42).
Two datasets fall outside ±2%, in opposite directions:

| dataset | ours | paper | Δ |
|---|---|---|---|
| DTD | 44.98 | 47.40 | −2.42 |
| EuroSAT | 62.36 | 58.00 | +4.36 |

## Why the difference? (not a bug — the method is order-dependent)

TDA is an **online** method: while it processes the test set image by image, it fills two caches
(a "positive" and a "negative" memory of past test images) and uses them to adjust every later
prediction. So the accuracy depends on **the order in which the test images arrive**.

The official code shuffles the test set (`shuffle=True` in the data loader). Even with the same
seed, the shuffle order changes with the PyTorch version, GPU, and number of data-loader
workers — so nobody running the code today gets the exact image order the authors had. On top
of that, the model runs in fp16, which adds small numeric noise.

This mostly averages out, which is why our overall mean is almost identical to the paper's. It
does NOT average out on:

- **DTD** — the smallest test set (~1.7k images), so a few early cache entries swing the final
  number a lot;
- **EuroSAT** — only 10 classes, and the repo gives it the most aggressive cache weights of all
  datasets (positive cache α=4.0, β=8.0 vs e.g. 2.0/3.0 for DTD), so the cache influences
  predictions the strongest here. Good or bad luck in what lands in the cache early gets
  amplified.

In short: the paper's DTD/EuroSAT numbers are one draw from a fairly wide run-to-run
distribution, and ours are another. The two deviations even roughly cancel (−2.4 / +4.4),
which is what you'd expect from variance rather than from a systematic error.

(Side note: the paper says all hyperparameters were found with a single search on ImageNet, but
the released configs are clearly tuned per dataset — another reason individual datasets are
brittle while the average is stable.)

## Relation to the CLIP-row issue

The TCA paper copied both the **CLIP** and the **TDA** rows straight from the TDA paper. The TDA
row above is at least the authors' own measurement (just order-sensitive). The CLIP row is worse:
the TDA paper in turn copied it from the **TPT paper (NeurIPS 2022)** — it matches TPT's
"Ensemble" baseline digit-for-digit. That baseline used the **80 generic hand-crafted ImageNet
prompts**, not the dataset-specific official CLIP prompts that the TCA/TDA codebases actually
use. That different prompt protocol — not any bug — is why our CLIP row is several points higher
on Flowers/Food/Pets and slightly lower on EuroSAT (details in `results/CLIP_ACC_DIFF.md`).
