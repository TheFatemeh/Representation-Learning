# Table 2 (CIFAR-100-C) — how we reproduced it, and every setting the paper left unstated

Table 2 of the TCA paper — *"Improvements over CLIP inference on CIFAR-100-C"* — reports
**Contrast, Snow, Brightness** at severities 1–5. The CLIP column is absolute top-1 accuracy;
the **EViT** and **Ours (TCA)** columns are the improvement (Δ) over CLIP. This file documents
exactly how we ran it and, more importantly, the long list of experimental choices the paper
never specifies — because if our numbers differ, the difference will live in these choices, and
we want them on the record.

## How we ran it

- **Scripts:** `scripts-t2/run_{clip,clip80,evit,tca}_cifar100c.sh`; assembled with
  `scripts-t2/build_table2.py`. Data via `scripts-t2/download_cifar100c.sh`. Full steps in
  `Table2-setup.md`.
- **Methods / settings — identical to our Table 1 runs:**
  - CLIP = pure zero-shot, no token pruning (`Ours-0.0`), no reservoir.
  - EViT R=0.9 = `clip_zeroshot.py --token_pruning EViT-0.1` (drop 0.1 at layers 3/6/9), no reservoir.
  - TCA R=0.9 ("Ours") = `runner.py --token_pruning Ours-0.035`.
  - `batch_size=1`, seed 1, ViT-B/16, deterministic.
- **Data:** the full 10 000-image CIFAR-100 test set per (corruption, severity), upsampled 32→224
  by the standard CLIP preprocess.
- **Outputs:** `results-t2/<METHOD>_cifar100c-<corruption>-<severity>.txt`.

## Specification comparison: our setting vs the paper (Table 2)

The paper specifies the experimental setup in only two places: §4.1 *Implementation Details*
(general, not CIFAR-specific) and the two-sentence §4.3 paragraph "Results on Various Severity
Levels." Everything below marked **"not stated"** is absent from both the main text and the
supplementary for CIFAR-100-C.

| Specification | Our setting | Paper (Table 2) | Stated in paper? |
|---|---|---|---|
| Backbone | CLIP ViT-B/16 | ViT-B/16 (§4.1: "ablations… on CLIP with ViT-B/16") | Stated generally; **not** restated for Table 2 — and the numbers contradict it (see below) |
| Visual preprocess | stock CLIP: Resize 224 (BICUBIC) + CenterCrop 224 (32→224 upscale) | — | **Not stated** |
| Effective input resolution | 224 px | — | **Not stated** |
| Prompt template | 18-template OpenAI CIFAR-100 ensemble (+ 80-prompt variant) | "official prompts" | Stated only as "official prompts" |
| Batch size | 1 | 1 | Stated |
| Augmentation | none | none | Stated |
| Test set per cell | full 10,000 images | — | **Not stated** |
| Corruptions | contrast, snow, brightness | contrast, snow, brightness | Stated |
| Severities | 1–5 | 1–5 | Stated |
| Seed / determinism | seed 1, deterministic | — | **Not stated** |
| Keep ratio R | R=0.9 (`Ours-0.035` / `EViT-0.1`) | — (main results use R=0.9) | **Not stated for this table** |
| Merge centers K | 4 (code default, first impl) → 2 (sweep) | 2 (§4.1: "We set K to 2") | Stated — **first-impl mismatch (4 vs 2)** |
| Merge:prune ratio α | code default | 2:1 | Stated in supp (Tab. 9) |
| Reservoir update | diversity-enforced (code defaults) | diversity-enforced best | Stated (Tab. 4) |
| Reservoir size M | 2 (EuroSAT) | — | **Not stated for CIFAR-100-C** |
| `scale` / β (layer scale) | 5 (EuroSAT) | — | **Not stated for CIFAR-100-C** |
| `lambd` / λ (correction wt) | 8 (EuroSAT) | — | **Not stated for CIFAR-100-C** |
| `beta` (temperature) | 7 (EuroSAT) | — | **Not stated for CIFAR-100-C** |
| GPU | A100 | RTX A6000 | Stated (HW only, no effect on accuracy) |

**Bottom line:** the paper's stated spec (ViT-B/16 + official prompts + bs1) is *our* spec, yet
our CLIP comes out ~2× higher. The discrepancy must live in one of the "not stated" rows —
preprocessing/resolution or, more likely, the actual backbone.

## The CLIP-baseline discrepancy: most likely a weaker backbone, not low resolution

Our honest CLIP is far above the paper's (e.g. contrast sev1 67.5% vs 31.9%; brightness ~67% vs
~41%). Two candidate explanations, both unstated:

1. **Low effective resolution.** A resolution sweep (`scripts-t2/run_clip_ressweep.sh` +
   `build_ressweep.py`) shows ~**16 px** best matches the paper's CLIP overall (mean abs error
   5.84 vs ~19–30 at other resolutions; snow@16 is nearly exact). But it gets *brightness* wrong:
   our 16px brightness declines with severity (38→29) while the paper's stays flat.

2. **A weaker backbone (most likely RN50).** Brightness is the gentlest corruption, so the
   brightness column ≈ the model's *clean* CIFAR-100 accuracy. The paper's brightness row is flat
   at **~41%** (41.00 / 41.44 / 41.83 / 41.12 / 38.10). CLIP clean CIFAR-100 zero-shot is
   **RN50 ≈ 41.6%**, ViT-B/32 ≈ 65%, ViT-B/16 ≈ 68.7%. A flat ~41% is the fingerprint of an
   **RN50-class backbone**, not the ViT-B/16 the paper claims. This explains all three columns
   (flat brightness, moderate snow decline, steep contrast collapse to 2.69%) better than the
   blur proxy, which only fit snow.

**Conclusion:** the paper's Table-2 CLIP baseline is *not reproducible from its own stated
protocol*. The flat ~41% brightness column points to an undocumented weaker backbone (RN50),
making the dramatic "Ours" gains partly an artifact of a much weaker baseline than the ViT-B/16
specified. We could not verify directly — this fork's loader hardcodes `--backbone ViT-B/16`, so
an RN50 run needs a code change — but the ~41% clean-accuracy fingerprint is strong. This is the
Table-1 "gain-vs-baseline-strength" finding in its sharpest form (cf. `summary/CLIP.md`).

## The prompt question (and why we run two CLIP baselines)

The paper does not state which prompt it used for CIFAR-100-C. From the Table-1 investigation we
know the paper's *cross-dataset* CLIP row was inherited from the TPT 80-prompt "Ensemble"
protocol rather than per-dataset prompts (see `summary/CLIP.md`). To be safe we report **both**:

1. **18-template OpenAI CIFAR-100 ensemble** (the headline baseline — all Δ's are computed
   against this). Defined in `TCA/datasets/cifar100c.py`; CLIP, EViT and TCA all read it, so the
   three methods share the same text classifier and the Δ's are apples-to-apples.
2. **80 hand-crafted ImageNet prompts** (`clip_zeroshot.py --imagenet-ensemble`) — the TPT
   "Ensemble" protocol, for comparison against the paper's absolute CLIP numbers.

## Settings the paper leaves unstated — what we chose and why

None of the following appears in the paper or its supplementary for CIFAR-100-C. We make each
choice explicit so a reviewer can see precisely where our pipeline is pinned.

| Unstated quantity | Paper | Our choice | Rationale |
|---|---|---|---|
| Prompt template | not given | 18-template OpenAI CIFAR-100 ensemble (+ 80-prompt as a second baseline) | Official CLIP CIFAR-100 templates; covered both plausible protocols. |
| `reservoir_size` M | not given | **2** (EuroSAT) | EuroSAT is the closest analog (see below). |
| `scale` (layer-scale temperature) | not given | **5** (EuroSAT) | same. |
| `lambd` λ (correction weight) | not given | **8** (EuroSAT) | same. |
| `beta` β (temperature) | not given | **7** (EuroSAT) | same. |
| keep ratio R | not given for this table | **R=0.9** (`Ours-0.035` / `EViT-0.1`) | matches the FLOP-matched setting used throughout Table 1. |
| α / merge:prune ratio | not given (supp default 2:1) | **code default** | hardcoded in the released code; we did not override it. |
| reservoir update strategy | Table 4 says diversity-enforced is best | **code default** (`--div`, `--reservoir-sim`, `--token_sim`, `--flag` all True) | the released defaults already select the diversity-enforced path. |

### Why EuroSAT is the template for the four TCA hyper-parameters

We had to pick *some* per-dataset config for `reservoir_size / scale / lambd / beta`, and the
repo ships one per published dataset. Among them **EuroSAT is the closest analog to
CIFAR-100-C**:

- **Low CLIP zero-shot accuracy** — both are regimes where CLIP is weak (CIFAR-100-C drops to
  single digits at high severity), so the reservoir's correction matters most.
- **Reliance on low-level visual cues** rather than fine object semantics.
- **Largest TCA gain over CLIP** of any published dataset — exactly the behaviour Table 2
  reports for CIFAR-100-C (the "Ours" Δ grows sharply with severity, e.g. +18.59% at Contrast
  sev 5).

So copying EuroSAT's tuned config is the most defensible starting point. It is recorded in
`TCA/configs/cifar100c.yaml`. If our "Ours" deltas come out below the paper's, the most likely
cause is that these four values were tuned per-dataset and CIFAR-100-C's true tuned values are
simply unknown to us.

## What to expect / how to read the result

`build_table2.py` prints our table next to the paper's. Interpretation hooks:
- The **EViT** column should be a small negative Δ at most severities (matches the paper's
  pattern: pruning tokens slightly hurts a content-sparse low-res input).
- The **Ours (TCA)** column should be positive and **grow with severity** — the headline claim.
  Whether the magnitude matches depends entirely on the unstated hyper-parameters above.
- If CLIP absolute differs from the paper, compare the 80-prompt baseline too — that, not the
  method, is the usual source of CLIP-row discrepancies (see `summary/CLIP.md`).
