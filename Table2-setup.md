# Table 2 Setup & Reproduction (CIFAR-100-C)

Reproduces **Table 2** of the TCA paper — *"Improvements over CLIP inference on CIFAR-100-C"* —
on CLIP ViT-B/16. The paper reports three corruptions (**Contrast, Snow, Brightness**) across
five severities; the CLIP column is absolute top-1 accuracy and the **EViT** / **Ours (TCA)**
columns are the improvement (Δ) over CLIP at that corruption+severity.

Everything for Table 2 is self-contained under `scripts-t2/` (scripts), `results-t2/` (per-run
outputs), and `output-t2/` (SLURM logs). The Table-1 pipeline under `scripts/`, `results/` is
untouched. Run on the TinyGPU/A100 cluster; GPU jobs are submitted with `sbatch`.

> Environment + CLIP assets are identical to Table 1 — see `SETUP.md` §1–§2 (conda env `TTA`,
> the `mkl<2024` / `numpy=1.26.4` / `transformers==4.46.2` pins, and `~/.cache/clip/ViT-B-16.pt`).

---

## 1. Download CIFAR-100-C

```bash
bash scripts-t2/download_cifar100c.sh
```

Pulls `CIFAR-100-C.tar` from Zenodo (~2.9 GB, all 19 corruptions + `labels.npy`) and extracts
**only** the three Table-2 corruptions plus labels into `TCA/data/CIFAR-100-C/`:

```
TCA/data/CIFAR-100-C/{contrast,snow,brightness}.npy   # each (50000,32,32,3) uint8
TCA/data/CIFAR-100-C/labels.npy                        # (50000,)
```

Each `.npy` stacks the 10 000-image CIFAR-100 **test** set at severities 1–5; severity `s` is
rows `[(s-1)*10000 : s*10000]`. No JSON split or symlink fixes are needed (unlike the §4
cross-dataset layout in `SETUP.md`) — the loader reads the arrays directly.

To reproduce beyond Table 2, edit the `tar -xf` line in the script to extract all corruptions
(`tar -xf "$TAR" -C "$DATA_ROOT"`) and add their names to the run scripts.

## 2. Code added for CIFAR-100-C

| File | Purpose |
|---|---|
| `TCA/datasets/cifar100c.py` *(new)* | `CIFAR100C` dataset (npy → PIL → CLIP `preprocess`), the 100 CIFAR-100 classnames in label order, and the **18-template** OpenAI CIFAR-100 prompt ensemble. |
| `TCA/configs/cifar100c.yaml` *(new)* | TCA hyper-parameters (copied from EuroSAT — see §5 / `summary/Table2.md`). |
| `TCA/utils.py` *(edited)* | `build_test_data_loader` gets a `cifar100c-<corruption>-<severity>` branch; `get_config_file` maps any `cifar100c-*` → `cifar100c.yaml`. |

No changes to `runner.py` or `clip_zeroshot.py` were needed: both already take the prompt
`template` from the dataset object, so CLIP / EViT / TCA all use the 18-template ensemble
automatically. The dataset name carries the corruption+severity through the existing
`--datasets` argument (e.g. `--datasets cifar100c-contrast-5`).

## 3. Run the experiments (submit on A100)

```bash
sbatch scripts-t2/run_clip_cifar100c.sh     # CLIP, 18-template  -> results-t2/CLIP_cifar100c-<c>-<s>.txt
sbatch scripts-t2/run_clip80_cifar100c.sh   # CLIP, 80-prompt     -> results-t2/CLIP80_cifar100c-<c>-<s>.txt
sbatch scripts-t2/run_evit_cifar100c.sh     # EViT R=0.9          -> results-t2/EViT_cifar100c-<c>-<s>.txt
sbatch scripts-t2/run_tca_cifar100c.sh      # TCA R=0.9 (Ours)    -> results-t2/TCA_R0.9_cifar100c-<c>-<s>.txt
```

Each script loops over Contrast/Snow/Brightness × severities 1–5 (15 runs each). Settings match
Table 1 exactly: CLIP = pure zero-shot (`Ours-0.0`), EViT = `--token_pruning EViT-0.1`,
TCA = `runner.py --token_pruning Ours-0.035` (= R=0.9). `batch_size=1`, seed 1, ViT-B/16.

**Runtime note:** TCA is `batch_size=1` with the reservoir → 3 × 5 × 10 000 = 150 000 forward
passes; its script requests `--time=24:00:00`. CLIP/EViT are much faster.

**Two CLIP baselines on purpose:** Table 1 showed the paper's CLIP row was inherited from the
TPT 80-prompt "Ensemble" protocol, *not* per-dataset prompts. For CIFAR-100-C the paper states
no prompt at all, so we run **both** the 18-template CIFAR ensemble (the headline baseline; all
Δ's are computed against it) **and** the 80 ImageNet prompts, and report both.

## 4. Assemble the table

```bash
python3 scripts-t2/build_table2.py
```

Parses `results-t2/*.txt`, computes `EViT−CLIP` and `TCA−CLIP` deltas (vs the 18-template CLIP),
and prints our table beside the paper's numbers, plus the CLIP 80-prompt absolutes. Flags any
missing/unparsed run.

## 5. Unstated settings — what we chose and why

The paper gives **no** CIFAR-100-C hyper-parameters anywhere (main text or supplementary). We
adopt the **EuroSAT** config (`reservoir_size 2, scale 5, lambd 8, beta 7`) verbatim and document
every assumption in `summary/Table2.md`. Short version: among the published datasets EuroSAT is
the closest analog to CIFAR-100-C (low CLIP zero-shot accuracy, low-level visual cues, largest
TCA gain over CLIP), so its tuned config is the most defensible starting point.
