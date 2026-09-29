# Reproducing TCA, and stress-testing its token reservoir

Work on *Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation*
(Wang et al., ICCV 2025) — a reproduction of its reported results, and an experiment testing
how robust the token reservoir the method depends on is to an early wrong entry.

All runs use CLIP ViT-B/16, batch size 1, seed 1, on the TinyGPU cluster.

## Where to start

| I want to… | Read |
|---|---|
| Reproduce the cross-dataset benchmark (Table 1) | [`SETUP.md`](SETUP.md) |
| Reproduce the CIFAR-100-C results (Table 2) | [`Table2-setup.md`](Table2-setup.md) |
| Run the reservoir contamination study | [`TCA-StressTest/README.md`](TCA-StressTest/README.md) and [`TCA-StressTest/StressTest.md`](TCA-StressTest/StressTest.md) |
| See how each baseline was run, and why our CLIP differs from the paper's | [`summary/`](summary/) |

## Layout

```
TCA/              reproduction code (fork of the official TCA repo) + run outputs in results/
TDA/              TDA baseline (official repo), used for one row of Table 1
TCA-StressTest/   instrumented fork for the contamination study
scripts/          Table 1 run scripts        results/      Table 1 outputs
scripts-t2/       Table 2 run scripts        results-t2/   Table 2 outputs
summary/          per-method notes on how each row was produced
```

Datasets are not committed; the download scripts in `scripts/` and
`TCA-StressTest/download_eurosat.sh` fetch them.
