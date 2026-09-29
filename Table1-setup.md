# Setup & Reproduction

Reproduces the cross-dataset (CD) rows of Table 1 of the TCA paper — **CLIP, TCA(R=0.9),
EViT(R=0.9), ToME(R=0.9), TDA**, plus a CLIP-via-TDA cross-check, on CLIP ViT-B/16.
Run on the TinyGPU cluster (NHR@FAU); GPU jobs are submitted with `sbatch`.

Results land in `results/` as `<METHOD>_<dataset>.txt` (e.g. `CLIP_eurosat.txt`,
`TCA_R0.9_eurosat.txt`, `TDA_eurosat.txt`). All run/download scripts live in `scripts/`;
SLURM logs are written to `output/`.

---

## 1. Environment

```bash
conda env create -f TCA/environment_clean.yaml
conda activate TTA
```

`environment_clean.yaml` leaves `numpy`/`transformers` unpinned, which resolve to versions
incompatible with `pytorch 2.1.0`. Fix them once (env is shared, so this persists):

```bash
# mkl 2025 removed a symbol pytorch 2.1 needs ("undefined symbol: iJIT_NotifyEvent");
# pytorch also needs llvm-openmp<16, so mkl must be 2023.x:
conda install -n TTA -y --solver=libmamba "mkl<2024" mkl-service      # -> mkl 2023.2.0, mkl-service 2.4.1
# torch 2.1 was compiled against numpy 1.x:
conda install -n TTA -y --solver=libmamba "numpy=1.26.4"
# transformers 4.57 needs torch>=2.2 ('register_pytree_node'); TCA doesn't use transformers directly:
pip install transformers==4.46.2          # the env's pip (TTA already activated above)
```

Sanity check on a GPU node: `python -c "import torch, clip, torchvision; print(torch.cuda.is_available())"`.

## 2. CLIP assets (TCA repo)

The TCA repo's `clip/` is missing the BPE vocab; the CLIP checkpoint is auto-downloaded by
`clip.load` on a node with internet (pre-stage it if compute nodes are offline):

```bash
wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz \
     -O TCA/clip/bpe_simple_vocab_16e6.txt.gz
mkdir -p ~/.cache/clip
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt \
     -O ~/.cache/clip/ViT-B-16.pt
```
(The TDA repo ships its own BPE file and reuses the same `~/.cache/clip/ViT-B-16.pt`.)

## 3. Download datasets

```bash
bash scripts/download_datasets.sh   # 10 CD datasets (incl. SUN397) + all CoOp split JSONs
bash scripts/download_sun.sh        # SUN397 images only (if you skipped it earlier; ~37 GB)
```

## 4. Data-layout fixes (so the loaders resolve)

Applied once after `download_datasets.sh` (already in place in this repo):

```bash
# splits collected into one folder; loaders updated to read TCA/splits/...
mkdir -p TCA/splits && mv TCA/split_zhou_*.json TCA/splits/ 2>/dev/null
sed -i 's#"split_zhou_#"splits/split_zhou_#' TCA/datasets/*.py

cd TCA/data
# caltech101: the .tar.gz is actually a zip containing a NESTED tar.gz -> extract both, then symlink
( cd caltech101 && unzip -o -q 101_ObjectCategories.tar.gz && cd caltech-101 && tar xzf 101_ObjectCategories.tar.gz )
ln -sfn caltech-101/101_ObjectCategories caltech101/101_ObjectCategories
# eurosat: split expects <Class>/..., images are under 2750/<Class>
( cd eurosat && for d in 2750/*/; do ln -sfn "2750/$(basename "$d")" "$(basename "$d")"; done )
# fgvc: loader wants fgvc/data/{images,variants.txt,...}
ln -sfn fgvc-aircraft-2013b/data fgvc/data
# ucf101: split expects <Class>/..., images are under UCF-101-midframes/<Class>
( cd ucf101 && for d in UCF-101-midframes/*/; do ln -sfn "UCF-101-midframes/$(basename "$d")" "$(basename "$d")"; done )
cd ../..
# oxford_flowers: loader image_dir edited to .../jpg (flat filenames in the split)
```

## 5. Reproduce Table 1 (submit on TinyGPU)

```bash
sbatch scripts/run_clip_zeroshot.sh   # CLIP        -> results/CLIP_<ds>.txt
sbatch scripts/run_clip_80prompts.sh  # CLIP from TPT
sbatch scripts/run_tca_repro.sh       # TCA R=0.9   -> results/TCA_R0.9_<ds>.txt   (token_pruning Ours-0.035)
sbatch scripts/run_evit.sh            # EViT R=0.9  -> results/EViT_R0.9_<ds>.txt   (clip_zeroshot + EViT-0.1)
sbatch scripts/run_tome.sh            # ToME R=0.9  -> results/ToME_R0.9_<ds>.txt   (clip_zeroshot + ToME-0.1)
sbatch scripts/run_gflops.sh          # GFLOPs all  -> results/GFLOPs.txt

(All run scripts already include `sun397` in their dataset list — once its images are downloaded,
these cover it too.)
```

## 6. TDA baseline (official repo, data symlinked from TCA — no re-download)

```bash
git clone --depth 1 https://github.com/kdiAAA/TDA.git TDA

# Build TDA/data as symlinks into TCA/data (TDA wants splits INSIDE each dataset dir + some
# different dir names: caltech-101, fgvc_aircraft, stanford_cars/cars_test, split_zhou_EuroSAT.json):
TCA=$(pwd)/TCA; T=$(pwd)/TDA/data; S=$TCA/splits; mkdir -p "$T"
mkdir -p "$T/caltech-101"; ln -sfn "$TCA/data/caltech101/caltech-101/101_ObjectCategories" "$T/caltech-101/101_ObjectCategories"; ln -sfn "$S/split_zhou_Caltech101.json" "$T/caltech-101/split_zhou_Caltech101.json"
mkdir -p "$T/dtd"; ln -sfn "$TCA/data/dtd/images" "$T/dtd/images"; ln -sfn "$S/split_zhou_DescribableTextures.json" "$T/dtd/split_zhou_DescribableTextures.json"
mkdir -p "$T/eurosat"; ln -sfn "$TCA/data/eurosat/2750" "$T/eurosat/2750"; ln -sfn "$S/split_zhou_EuroSat.json" "$T/eurosat/split_zhou_EuroSAT.json"
ln -sfn "$TCA/data/fgvc/fgvc-aircraft-2013b/data" "$T/fgvc_aircraft"
mkdir -p "$T/food-101"; ln -sfn "$TCA/data/food-101/images" "$T/food-101/images"; ln -sfn "$S/split_zhou_Food101.json" "$T/food-101/split_zhou_Food101.json"
mkdir -p "$T/oxford_flowers"; ln -sfn "$TCA/data/oxford_flowers/jpg" "$T/oxford_flowers/jpg"; ln -sfn "$S/split_zhou_OxfordFlowers.json" "$T/oxford_flowers/split_zhou_OxfordFlowers.json"
mkdir -p "$T/oxford_pets"; ln -sfn "$TCA/data/oxford_pets/images" "$T/oxford_pets/images"; ln -sfn "$S/split_zhou_OxfordPets.json" "$T/oxford_pets/split_zhou_OxfordPets.json"
mkdir -p "$T/stanford_cars"; ln -sfn "$TCA/data/cars_test" "$T/stanford_cars/cars_test"; ln -sfn "$S/split_zhou_StanfordCars.json" "$T/stanford_cars/split_zhou_StanfordCars.json"
mkdir -p "$T/ucf101"; ln -sfn "$TCA/data/ucf101/UCF-101-midframes" "$T/ucf101/UCF-101-midframes"; ln -sfn "$S/split_zhou_UCF101.json" "$T/ucf101/split_zhou_UCF101.json"
mkdir -p "$T/sun397"; ln -sfn "$TCA/data/sun397/SUN397" "$T/sun397/SUN397"; ln -sfn "$S/split_zhou_SUN397.json" "$T/sun397/split_zhou_SUN397.json"

# CLIP-via-TDA configs (both caches OFF -> pure CLIP through TDA's pipeline):
mkdir -p TDA/configs_clip
for d in caltech101 dtd eurosat fgvc food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101; do
    sed 's/enabled: True/enabled: False/g' TDA/configs/$d.yaml > TDA/configs_clip/$d.yaml
done

sbatch scripts/run_tda.sh             # TDA          -> results/TDA_<ds>.txt
sbatch scripts/run_clip_via_tda.sh    # CLIP via TDA -> results/CLIPviaTDA_<ds>.txt
```
Note: `tda_runner.py` was patched with a guarded `wandb.init(mode="disabled")` (its `wandb.log`
is called every image) and a GFLOPs print; the run scripts also set `WANDB_MODE=disabled`.

## 7. SUN397 (convenience scripts)

`sun397` is already in every run script above, so steps 5–6 cover it once its images are
downloaded. If you ran the other 9 datasets first and only need to add SUN397:

```bash
sbatch scripts/run_sun_all.sh   # all 6 methods on SUN397 only -> results/*_sun397.txt
sbatch scripts/run_sun_tda.sh   # just TDA on SUN397
```

---

### Notes / findings
- `results/CLIP_ACC_DIFF.md` — why our CLIP differs from the paper's CLIP row (the paper's CLIP
  baseline is inconsistent with its own EViT/ToME rows and isn't reproducible from TCA's *or*
  TDA's released code; verified prompts are the OpenAI official ones).
- GFLOPs are content-independent constants per method (CLIP/TDA 17.58, EViT 15.41, ToME 15.30,
  TCA 15.71); the TCA value is slightly above the paper's 15.45 because the code hardcodes K=4
  merge centers vs the paper's K=2.
