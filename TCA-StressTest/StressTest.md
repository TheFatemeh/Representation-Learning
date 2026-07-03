# TCA Reservoir-Contamination Stress Test — File Map

What we added or changed inside `TCA-StressTest/` to run the reservoir-contamination
experiment on EuroSAT (CLIP ViT-B/16, `Ours-0.035` = R=0.9). Only the files that matter for
the test are listed. Plain-language change log is in `IMPLEMENTATION_NOTES.md`.

```
TCA-StressTest/
├── runner.py                       # CHANGED. Main eval loop. Added: new CLI flags
│                                   #   (--update_rule, --reservoir_mode, --reservoir_size,
│                                   #   --target_class, --poison_path, --out_dir, --seed,
│                                   #   --visualize_mask); the 2 missing update rules (FIFO,
│                                   #   uncertainty); the 3 reservoir modes (clean/empty/
│                                   #   seeded); per-sample / per-timestep / scalar logging;
│                                   #   poison identification (empty mode).
├── utils.py                        # CHANGED. Two 1-line fixes so the new per-entry tags
│                                   #   don't break the token-similarity reads
│                                   #   (item[-1] -> item[2]). Data loader/order left as stock.
│
├── scripts/                        # NEW. All run scripts. Submit each FROM THE REPO ROOT.
│   ├── smoke_clean.sbatch          #   1 clean run; smoke test that TCA reproduces the paper.
│   ├── sweep_clean.sbatch          #   16 clean baselines: 4 rules x M{1,2,3,5}
│   │                               #     (M=2 = paper Table-4 check; M{1,3,5} = the baselines).
│   ├── identify.sbatch             #   Stage 2: 1 empty-reservoir run -> finds the 10 poisons.
│   ├── sweep_poison_M1.sbatch      #   Stage 3: 40 poisoned runs at M=1 (4 rules x 10 classes).
│   ├── sweep_poison_M3.sbatch      #   Stage 3: 40 poisoned runs at M=3.
│   └── sweep_poison_M5.sbatch      #   Stage 3: 40 poisoned runs at M=5.
│
├── results/                        # GENERATED. All experiment outputs.
│   ├── runs/<key>/                 #   one folder per run; key = rule-<r>__M<M>__target-<c|none>
│   │   ├── scalar__<key>.json      #     overall accuracy + avg GT-confidence + run metadata
│   │   ├── persample__<key>.csv    #     per image: true/pred class, correct?, conf on pred & GT
│   │   └── ts__<key>.csv           #     per timestep: contaminants per buffer, seeded poison,
│   │                               #       source-in-target count
│   ├── identify/
│   │   ├── poisons.csv             #   the 10 chosen poisons (target c, source c', conf, id)
│   │   └── poison_seeds/poison_c{c}.pt  # serialized seed tensors injected by the seeded runs
│   ├── clean_baselines.csv         #   tidy table of all clean runs
│   └── clean_validation.md         #   clean results + M=2 vs paper Table 4
│
├── clip/bpe_simple_vocab_16e6.txt.gz  # SETUP. CLIP tokenizer vocab, copied in from sibling
│                                   #   TCA/clip/ (was missing from the clone; every run needs it).
│
├── IMPLEMENTATION_NOTES.md         # NEW. Plain-language log of every change, in order.
├── StressTest.md                   # NEW. This file.
│
└── aggregate.py                    # NOT YET CREATED (Stage 4): rolls results/ into the tidy
                                    #   scalar/disaggregated tables + the per-rule contamination plot.
```

## How to run (in order)

```bash
# from the TCA-StressTest repo root:
sbatch scripts/sweep_poison_M1.sbatch   # 40 runs, M=1
sbatch scripts/sweep_poison_M3.sbatch   # 40 runs, M=3
sbatch scripts/sweep_poison_M5.sbatch   # 40 runs, M=5
# then, once all finish:
# python3 aggregate.py                  # (Stage 4, to be added)
```

Stages 1 (instrument), 2 (identify), and the clean baselines are already done.
```
