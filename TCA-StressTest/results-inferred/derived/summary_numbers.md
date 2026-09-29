# Summary numbers


**Contamination floor (clean runs, steady state):** 0.194 of the reservoir overall; per rule fifo 0.424, similarity 0.162, diversity 0.094, uncertainty 0.097.

**Effect sizes:** of 120 poisoned runs, 32 moved target accuracy by >1 pp, 22 by >5 pp, and 15 by >10 pp.

**FIFO diagnostic:** CONSISTENT WITH FAST EVICTION: FIFO evicts the seed within a few dozen timesteps, so it is read out on very few predictions and most runs show no flips. No evidence of a read-path bug.

## provenance

```json
{
  "n_poisoned_runs": 120,
  "n_clean_runs": 16,
  "missing_poisoned": [],
  "missing_clean": [],
  "extra_clean_outside_grid": [
    "diversity/M2",
    "fifo/M2",
    "similarity/M2",
    "uncertainty/M2"
  ],
  "pairing_check": "passed (136 runs, 8100 samples, reference rule-diversity__M1__target-0)",
  "seeds_present": [
    1
  ]
}
```

## contamination floor clean

```json
{
  "per_rule_steady_state_fraction": {
    "fifo": 0.42401262002743484,
    "similarity": 0.16177119341563787,
    "diversity": 0.09408504801097395,
    "uncertainty": 0.09737064471879288
  },
  "overall_steady_state_fraction": 0.1943098765432099,
  "note": "fraction of reservoir slots holding a wrong-class entry, 2nd half of the stream"
}
```

## effect counts

```json
{
  "|delta target acc| > 1pp": 32,
  "|delta target acc| > 5pp": 22,
  "|delta target acc| > 10pp": 15,
  "n_runs": 120,
  "n_runs_target_acc_decreased": 46,
  "n_runs_target_acc_increased": 31,
  "n_runs_target_acc_unchanged": 43
}
```

## survival

```json
{
  "per_rule": {
    "fifo": {
      "median_t_evict_evicted_only": 30.0,
      "median_residency_censored_as_max": 30.0,
      "n_censored": 0,
      "n_runs": 30
    },
    "similarity": {
      "median_t_evict_evicted_only": 314.5,
      "median_residency_censored_as_max": 387.5,
      "n_censored": 2,
      "n_runs": 30
    },
    "diversity": {
      "median_t_evict_evicted_only": 585.0,
      "median_residency_censored_as_max": 924.5,
      "n_censored": 3,
      "n_runs": 30
    },
    "uncertainty": {
      "median_t_evict_evicted_only": 585.0,
      "median_residency_censored_as_max": 924.5,
      "n_censored": 3,
      "n_runs": 30
    }
  },
  "longest_surviving_rule": "diversity and uncertainty (tied)",
  "longest_surviving_tied_rules": [
    "diversity",
    "uncertainty"
  ],
  "ratio_longest_to_fifo_median_residency": 30.816666666666666,
  "note": "residency = t_evict, or 8100 (stream length) for censored runs; the ratio is a lower bound because censored runs are truncated"
}
```

## damage after eviction

```json
{
  "n_runs_considered": 24,
  "mean_frac_flips_after_evict": 0.35792898535996126,
  "median_frac_flips_after_evict": 0.27446954972465176,
  "definition": "a flip at timestep >= t_evict is post-eviction, because runner.py updates the reservoir before reading it out"
}
```

## attractor direction

```json
{
  "fifo": {
    "n_fell": 0,
    "n_rose": 20,
    "n_unchanged": 10,
    "mean_change": 0.00020987654320988152,
    "sum_of_drops": 0.0,
    "sum_of_rises": 0.0062962962962964455
  },
  "similarity": {
    "n_fell": 15,
    "n_rose": 10,
    "n_unchanged": 5,
    "mean_change": -0.007209876543209887,
    "sum_of_drops": -0.23271604938271634,
    "sum_of_rises": 0.016419753086419686
  },
  "diversity": {
    "n_fell": 15,
    "n_rose": 9,
    "n_unchanged": 6,
    "mean_change": -0.00833333333333334,
    "sum_of_drops": -0.2641975308641978,
    "sum_of_rises": 0.014197530864197512
  },
  "uncertainty": {
    "n_fell": 15,
    "n_rose": 11,
    "n_unchanged": 4,
    "mean_change": -0.008592592592592601,
    "sum_of_drops": -0.26333333333333353,
    "sum_of_rises": 0.005555555555555501
  },
  "ALL": {
    "n_fell": 45,
    "n_rose": 50,
    "n_unchanged": 25,
    "mean_change": -0.005981481481481487,
    "sum_of_drops": -0.7602469135802474,
    "sum_of_rises": 0.042469135802469145,
    "largest_change_key": "rule-similarity__M3__target-1",
    "largest_change": -0.0935802469135803
  }
}
```

## h5 recruitment

```json
{
  "n_runs_with_recruitment": 92,
  "n_runs": 120,
  "frac_runs_with_recruitment": 0.7666666666666667,
  "n_recruitment_and_source_acc_improved": 27,
  "n_recruitment_and_source_acc_worsened": 15,
  "mean_d_source_acc_when_recruited": 0.7577294685990339,
  "mean_d_source_acc_when_not_recruited": -0.07142857142857295,
  "definition": "recruitment = a source-class entry in the targeted buffer other than the seed itself (source_in_target - seeded_poison > 0)"
}
```

## fifo diagnostic

```json
{
  "median_t_evict_evicted_only": 30.0,
  "median_residency": 30.0,
  "mean_n_flip": 1.8,
  "max_n_flip": 23.0,
  "n_runs_zero_flips": 16,
  "n_runs": 30,
  "n_runs_resident_over_100_steps": 7,
  "n_of_those_with_zero_flips": 1,
  "verdict": "CONSISTENT WITH FAST EVICTION: FIFO evicts the seed within a few dozen timesteps, so it is read out on very few predictions and most runs show no flips. No evidence of a read-path bug."
}
```

## flip counts by rule

```json
{
  "fifo": {
    "mean_n_flip": 1.8,
    "n_runs_zero_flips": 16,
    "n_runs": 30
  },
  "similarity": {
    "mean_n_flip": 185.5,
    "n_runs_zero_flips": 7,
    "n_runs": 30
  },
  "diversity": {
    "mean_n_flip": 119.53333333333333,
    "n_runs_zero_flips": 4,
    "n_runs": 30
  },
  "uncertainty": {
    "mean_n_flip": 115.8,
    "n_runs_zero_flips": 4,
    "n_runs": 30
  }
}
```

## spearman fig3

```json
{
  "rho_t_evict": -0.3998667681169423,
  "n_t_evict": 112,
  "rho_excess_contam": -0.5393498036544964,
  "n_excess_contam": 120
}
```

## exemplars

```json
{
  "default": {
    "key": "rule-similarity__M3__target-1",
    "rule": "similarity",
    "M": 3,
    "target_class": 1,
    "source_class": 2,
    "t_evict": null,
    "censored": true,
    "d_target_acc": -76.33333333333334,
    "d_source_acc": 3.333333333333336,
    "n_flip_neg": 871,
    "n_flip_pos": 202,
    "recruit_max": 2,
    "cum_gap_at_evict": NaN,
    "cum_gap_final": 669.0,
    "gap_widened_after_evict": null
  },
  "moderate": {
    "key": "rule-similarity__M5__target-8",
    "rule": "similarity",
    "M": 5,
    "target_class": 8,
    "source_class": 2,
    "t_evict": 2997,
    "censored": false,
    "d_target_acc": -7.066666666666663,
    "d_source_acc": -0.4444444444444464,
    "n_flip_neg": 146,
    "n_flip_pos": 134,
    "recruit_max": 2,
    "cum_gap_at_evict": 29.0,
    "cum_gap_final": 12.0,
    "gap_widened_after_evict": false
  },
  "null": {
    "key": "rule-diversity__M1__target-0",
    "rule": "diversity",
    "M": 1,
    "target_class": 0,
    "source_class": 6,
    "t_evict": 199,
    "censored": false,
    "d_target_acc": 0.0,
    "d_source_acc": 0.0,
    "n_flip_neg": 0,
    "n_flip_pos": 0,
    "recruit_max": 1,
    "cum_gap_at_evict": 0.0,
    "cum_gap_final": 0.0,
    "gap_widened_after_evict": false
  }
}
```

## flip pool

```json
{
  "n_flip_events_total": 12679,
  "n_in_evicted_runs": 9539,
  "n_in_censored_runs": 3140,
  "frac_after_eviction": 0.43903973162805326
}
```

## largest effect run sinks

```json
[
  {
    "run": "rule-similarity__M3__target-1",
    "source_class": "2 HerbaceousVegetation",
    "n_lost": 688,
    "n_lost_to_source_class": 331,
    "frac_lost_to_source_class": 0.4811046511627907,
    "largest_sink": "2 HerbaceousVegetation",
    "largest_sink_count": 331
  },
  {
    "run": "rule-uncertainty__M1__target-1",
    "source_class": "2 HerbaceousVegetation",
    "n_lost": 445,
    "n_lost_to_source_class": 264,
    "frac_lost_to_source_class": 0.5932584269662922,
    "largest_sink": "2 HerbaceousVegetation",
    "largest_sink_count": 264
  },
  {
    "run": "rule-diversity__M1__target-1",
    "source_class": "2 HerbaceousVegetation",
    "n_lost": 445,
    "n_lost_to_source_class": 264,
    "frac_lost_to_source_class": 0.5932584269662922,
    "largest_sink": "2 HerbaceousVegetation",
    "largest_sink_count": 264
  }
]
```

## figure params

```json
{
  "fig2_symmetric_vmax_pp": 77.0
}
```

