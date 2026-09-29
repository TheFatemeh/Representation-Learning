# Analysis: reservoir-contamination stress test

Two files, split by what they contain:

| file | holds |
|---|---|
| `make_report_assets.py` | the computation — parsing runs, deriving metrics, drawing figures, emitting tables. No report prose. |
| `descriptions.py` | every word that appears in the output — figure titles, table captions, and all of `descriptions.md`. No analysis. |

Together they turn the raw per-run outputs of `runner.py` into every table, figure and number used
in the write-up. **Read-only** with respect to `results/`: all output goes to `results-inferred/`.

**To find out what an output file is**, look it up in `results-inferred/descriptions.md` — it opens
with an index of every generated file, then gives each one a caption, a description of how it was
computed and what is visible in it, and a note on what it does not show. To change wording, edit
`descriptions.py`; the lookups are keyed by output filename (`FIG_TEXT`, `TABLE_CAPTION`,
`ASSET_INDEX`).

## Running it

```bash
# this repo's env has an MKL/OpenMP conflict that surfaces when numpy is imported alongside
# matplotlib; MKL_THREADING_LAYER=GNU avoids it
MKL_THREADING_LAYER=GNU python analysis/make_report_assets.py

python analysis/make_report_assets.py --dry-run       # inventory + pairing check, writes nothing
python analysis/make_report_assets.py --reuse-cache   # skip re-parsing ~2.2M raw rows (~25s -> ~7s)
python analysis/make_report_assets.py --exemplar diversity,5,6   # different Fig 4 run
```

Flags: `--raw-dir`, `--out-dir`, `--exemplar rule,M,target_class`, `--reuse-cache`, `--dry-run`.
Requires numpy, pandas, matplotlib only (Spearman ρ is computed from average ranks, no scipy).

## What it does

1. **Discovers** runs by walking `--raw-dir` and reading each `scalar__*.json`; poisoned vs clean is
   decided by the `mode` field, never by the directory name. Missing runs are warned about and
   skipped, not fatal.
2. **Checks pairing** — every run must share the same `sample_id`/`true_class` sequence. If any run
   differs, all paired comparisons (Figs 2, 3, 4, 6, 7 and T2, T3) are skipped with a loud warning
   and the rest still builds.
3. **Collapses** each poisoned run to one row of `derived/run_level.csv` (survival, contamination,
   recruitment, compounding, prediction flips, accuracy deltas), cached along with
   `derived/flip_events.csv` for `--reuse-cache`.
4. **Emits** figures (300 dpi PNG + vector PDF), tables (`.csv`/`.tex`/`.md`), summary numbers, and
   `descriptions.md` — whose wording is generated *from the computed values*, so it states what the
   data show rather than what was hypothesised. The `_*_sentence()` helpers in `descriptions.py`
   each choose their claim by testing the numbers; if a hypothesised pattern is absent, the helper
   that covers it says so rather than asserting it.

## Output layout

```
results-inferred/
├── derived/    run_level.csv, flip_events.csv, summary_numbers.{json,md}, file_inventory.txt
├── tables/     T1–T4, T3b, A1–A3  (each as .csv / .tex / .md)
├── figures/    fig1–fig7 (+ fig4b/fig4c appendix exemplars), .png and .pdf
├── descriptions.md
└── run_log.txt
```

## Conventions worth knowing

- **Contaminant** = a stored entry whose *ground-truth* label differs from the buffer it sits in.
  Ground truth is an analysis-time oracle; TCA files entries by predicted class and never sees it.
- **Censoring**: a seed that survives all 8100 samples has no `t_evict`. Such runs are right-
  censored (`censored=True`, `residency=8100`), shown as `—` in T1, drawn in a separate lane in
  Fig 3, and excluded from "evicted only" medians. They are never silently dropped.
- **Post-eviction** means `timestep >= t_evict`: `runner.py` updates the reservoir *before* reading
  it out, so the prediction made at `t_evict` is already poison-free.
- **Recruitment** excludes the seed itself (`source_in_target - seeded_poison`), since the seed's
  own true label is the source class and would otherwise count as recruitment.
- Each metric is defined in exactly one function (`evict_time`, `contamination_metrics`,
  `recruitment_metrics`, `compounding_metrics`, `prediction_metrics`, `class_accuracy`); figures and
  tables all read those definitions rather than recomputing.
