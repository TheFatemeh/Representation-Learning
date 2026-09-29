# Reservoir Contamination Stress Test (TCA)

An instrumented fork of the official TCA code
([Jo-wang/TCA](https://github.com/Jo-wang/TCA)) used to test how robust TCA's
domain-aware token reservoir is to a wrong entry admitted early in the test stream.

TCA files each test sample into the buffer of its **predicted** class, so a confident
mistake is stored as though it were a correct example, and a training-free method cannot
correct it later. This fork measures what that costs, on EuroSAT.

## What was added to the stock code

- The two update rules missing from the release (`fifo`, `uncertainty`), alongside the two
  it ships (`similarity`, `diversity`), behind one `--update_rule` flag.
- Three reservoir modes (`--reservoir_mode`): `clean` (stock behaviour), `empty` (store
  nothing; used to identify poisons), `seeded` (preload one poison at *t*=0).
- Per-sample, per-timestep and per-run logging of accuracy, confidence and reservoir
  contamination.
- Fixes required to make the stock code run at all (see `StressTest.md`).

## Running it

```bash
# 1. data
bash download_eurosat.sh

# 2. find one poison per class (empty reservoir)
sbatch scripts/identify.sbatch

# 3. baselines and poisoned sweeps
sbatch scripts/sweep_clean.sbatch        # 4 rules x M in {1,2,3,5}
sbatch scripts/sweep_poison_M1.sbatch    # 40 runs, M=1
sbatch scripts/sweep_poison_M3.sbatch    # 40 runs, M=3
sbatch scripts/sweep_poison_M5.sbatch    # 40 runs, M=5

# 4. tables and figures
python3 aggregate.py                     # -> results/tables/
python3 analysis/make_report_assets.py   # -> results-inferred/
```

133 runs in total. Every run writes to `results/runs/<rule>__<M>__<target>/`.

## Layout

See **`StressTest.md`** for the annotated file map, the output schema, and what each
script and table contains.
