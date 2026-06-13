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
