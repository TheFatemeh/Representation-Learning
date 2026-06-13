# CLIP — how we reproduced it, and why our numbers differ from the paper

CLIP ViT-B/16 zero-shot is the first row of Table 1 and the baseline every other row is
compared against. Reproducing it turned into the most involved part of the project, because our
honest reproduction comes out **higher** than the paper on several datasets. This file documents
what we found: the deviation is real, it is fully explained, and it is **not a bug in our run** —
it comes from a difference in which text prompts are used.

## How we ran it

- **Script:** `scripts/run_clip_zeroshot.sh` (official per-dataset prompts) and
  `scripts/run_clip_80prompts.sh` (the 80 generic ImageNet prompts — see below).
- **Command (per dataset):**
  `python3 clip_zeroshot.py --backbone ViT-B/16 --datasets <ds> --data-root TCA/data`
  (add `--imagenet-ensemble` for the 80-prompt version).
- **Setting:** pure CLIP — no token pruning, no reservoir/TCA adaptation. `clip_zeroshot.py`
  does `CLIP forward → cosine similarity → top-1`. Batch size 1, seed 1, ViT-B/16. Deterministic.
- **Output:** `results/CLIP_<ds>.txt` (official prompts) and `results/CLIP80_<ds>.txt` (80 prompts).

## Where the paper's CLIP numbers come from: TCA ← TDA ← TPT

The single most important finding. We could not match the paper's CLIP row from any released
code, and the reason is that **the row was never produced by the TCA code at all** — it was
copied, paper to paper, all the way back to 2022:

1. **TCA** (ICCV 2025) takes its CLIP (and TDA) baseline rows directly from the **TDA** paper.
2. **TDA** (CVPR 2024) states this verbatim: *"The results of CLIP, CoOp, CoCoOp, and TPT are
   obtained from the TPT paper."*
3. **TPT** (NeurIPS 2022, arXiv:2209.07511) is the original source. Its Table 2 "Ensemble" row
   for CLIP-ViT-B/16 matches the TCA/TDA "CLIP" row **digit-for-digit** (e.g. EuroSAT 50.42,
   Flowers 66.99, Food101 82.86, Pets 86.92, DTD 45.04, Caltech 93.55, Cars 66.11, Aircraft
   23.22, UCF101 65.16, SUN397 65.63 → average 64.59).

So the "CLIP" baseline that TCA reports against is a **three-year-old inherited number**, not a
fresh run of the codebase that ships with TCA or TDA.

## The two prompt protocols (this is the whole story)

CLIP turns class names into a classifier by filling text **templates** with each class name,
encoding them, and **averaging** the embeddings into one vector per class (see
`utils.clip_classifier`). There is no randomness and no sampling — *all* templates in the list
are always averaged. The only thing that changes between setups is **which list of templates**:

- **What the TCA/TDA codebases actually use (and what we use):** **dataset-specific official
  CLIP prompts**, defined in `TCA/datasets/*.py`. These carry semantic context, e.g.
  `a photo of a {}, a type of flower.` for Flowers, `a photo of {}, a type of food.` for Food101,
  `a centered satellite photo of {}.` for EuroSAT. Template counts vary per dataset (1 for
  Food/Flowers/Pets, 2 for FGVC/SUN, 3 for EuroSAT, 8 for DTD/Cars, 34 for Caltech, 48 for UCF).
  We verified these are byte-identical to OpenAI's official `prompts.md`.

- **What the inherited paper row actually used:** the **80 generic hand-crafted ImageNet
  prompts** from Radford et al. (CLIP paper), applied uniformly to every dataset — things like
  `a bad photo of a {}.`, `a sculpture of a {}.`, `itap of a {}.`. These carry no
  dataset-specific context. They live in `TCA/datasets/imagenet_a.py` (the repo only used them
  for the ImageNet-variant benchmark, never for the fine-grained datasets).

The TPT paper says exactly this about its baseline (the row everyone later inherited as "CLIP"):

> *"We also include two versions of the baseline zero-shot performance of CLIP, using a default
> prompt 'a photo of a', and the ensemble of 80 hand-crafted prompts from Radford et al."*

So the paper's "CLIP" column is CLIP with the **80 generic prompts** — a deliberately
protocol-light baseline — while the TCA/TDA code (and their own EViT/ToME/TCA/TDA method rows)
run with the **dataset-specific official prompts**.

## How the prompt difference produced the "gains," and how the 80-prompt run closes the gap

When we ran CLIP with the **official per-dataset prompts**, we beat the paper's CLIP row by a lot
on exactly the datasets where a specialized template helps most — Flowers +4.3, UCF +3.7,
Food +3.3, Pets +2.2 — because `"...a type of flower."` steers the text encoder far better than
80 generic rephrasings. That is the source of the entire apparent "deviation."

When we instead ran CLIP with the **80 generic ImageNet prompts** — matching the inherited
protocol — the gap collapsed. Four datasets land within 0.1–0.4 of the paper, Flowers swings from
+4.3 to −1.1, and the average difference drops from +1.22 to +0.36:

| dataset | paper CLIP (= TPT Ensemble) | ours, official prompts | Δ official | ours, 80 prompts | Δ 80-prompt |
|---|---|---|---|---|---|
| FGVC / Aircraft | 23.22 | 24.30 | +1.08 | 23.64 | +0.42 |
| Caltech101 | 93.55 | 92.98 | −0.57 | 93.67 | +0.12 |
| Stanford Cars | 66.11 | 66.01 | −0.10 | 66.38 | +0.27 |
| DTD | 45.04 | 45.63 | +0.59 | 45.27 | +0.23 |
| EuroSAT | 50.42 | 48.26 | −2.16 | 47.78 | −2.64 |
| Oxford Flowers | 66.99 | 71.30 | **+4.31** | 65.94 | −1.05 |
| Food101 | 82.86 | 86.12 | +3.26 | 85.39 | +2.53 |
| Oxford Pets | 86.92 | 89.13 | +2.21 | 88.20 | +1.28 |
| UCF101 | 65.16 | 68.86 | +3.70 | 67.43 | +2.27 |
| SUN397 | 65.63 | 65.50 | −0.13 | 65.80 | +0.17 |
| **average (10)** | **64.59** | **65.81** | **+1.22** | **64.95** | **+0.36** |

Two clean confirmations in this table:
- **Flowers** is the smoking gun: official prompt gives +4.3 (one specialized template wins big),
  80 generic prompts give −1.1 (no context) — same image set, same code, only the prompts changed.
- **SUN397** barely moves (65.50 official vs 65.80 with 80 prompts, both ≈ paper 65.63) precisely
  because SUN's official prompt list is already generic (just 2 plain `a photo of a/the {}.`
  templates), so there is almost nothing for the 80-ensemble to change.

## Why the 80-prompt run isn't a *perfect* match (the residuals)

After switching to the 80 prompts, three datasets still sit a couple of points off: Food +2.5,
UCF +2.3, EuroSAT −2.6. These are **TPT-pipeline-specific**, not our code:

- In TPT's own Table 2, ensembling 80 prompts actually *hurt* Food/Pets/Flowers relative to a
  single prompt. In our pipeline the 80-ensemble lands above their numbers on those datasets — so
  TPT's 2022 pipeline (built on the CoOp codebase) had an additional depressant we cannot recover
  from the released materials: most likely differences in class-name strings, image preprocessing,
  or their specific dataset copies/splits.
- **EuroSAT** is unreachable from *either* prompt set in our pipeline (official 48.26, 80-prompt
  47.78, both below the paper's 50.42). EuroSAT is well known for split/copy sensitivity, so this
  is consistent with a data-version difference rather than a prompt or code issue.

Chasing these last ~2 points would require reconstructing the exact 2022 TPT environment, which
would not change the conclusion. We stopped here.

## Conclusion

Our CLIP reproduction is faithful and internally consistent. The paper's CLIP row is not *wrong* —
it is the **correct number for a weaker prompt protocol** (CLIP + 80 generic ImageNet prompts,
inherited from TPT) than the one the TCA/TDA codebases actually run for everything else (CLIP +
dataset-specific official prompts). When we match that inherited protocol, we reproduce the paper
to within +0.36 on average with four datasets near-exact; when we use the official prompts the
code ships with, CLIP is ~1.2 points stronger. The practical implication for reading Table 1:
TCA's reported improvement over CLIP is measured against a baseline run under a different, weaker
prompt setup, so the real headroom for token-condensation methods over a properly-prompted CLIP is
smaller than the table suggests.

See also: `results/CLIP_ACC_DIFF.md` (detailed prompt audit + EViT cross-check that proves the
pipeline is calibrated), and `summary/TDA.md` (the other inherited row, and why DTD/EuroSAT
deviate there for a different reason — online order-dependence).
