# Clean TCA baselines (EuroSAT, ViT-B/16, Ours-0.035 / R=0.9, seed 1)

Source: results/runs/<key>/scalar__*.json (job 1721578). Order/seed match the official repo.

| rule | M | acc | avg_conf_gt |
|---|---|---|---|
| diversity | 1 | 67.65 | 0.459 |
| diversity | 2 | 70.09 | 0.553 |
| diversity | 3 | 68.51 | 0.599 |
| diversity | 5 | 69.62 | 0.645 |
| fifo | 1 | 49.11 | 0.471 |
| fifo | 2 | 53.72 | 0.516 |
| fifo | 3 | 56.10 | 0.537 |
| fifo | 5 | 59.26 | 0.567 |
| similarity | 1 | 56.47 | 0.460 |
| similarity | 2 | 69.07 | 0.550 |
| similarity | 3 | 70.90 | 0.612 |
| similarity | 5 | 71.77 | 0.666 |
| uncertainty | 1 | 67.65 | 0.459 |
| uncertainty | 2 | 70.32 | 0.554 |
| uncertainty | 3 | 68.48 | 0.598 |
| uncertainty | 5 | 69.43 | 0.643 |

## M=2 vs paper Table 4 (EuroSAT)

| rule | ours | paper | diff |
|---|---|---|---|
| diversity | 70.09 | 70.43 | -0.34 |
| fifo | 53.72 | 50.28 | +3.44 |
| similarity | 69.07 | 68.48 | +0.59 |
| uncertainty | 70.32 | 70.20 | +0.12 |
