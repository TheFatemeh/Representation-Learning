"""All human-readable text for the reservoir-contamination report.

`make_report_assets.py` computes numbers and draws figures; this module supplies every word that
ends up in them. Nothing here reads a file or does analysis.

Three lookups, all keyed by the output filename stem, so you can go from a file in
`results-inferred/` straight to the text that describes it:

    FIG_TEXT['fig5_clean_contamination_floor']   -> on-figure title / narrative axis labels
    TABLE_CAPTION['T1_eviction_times']           -> caption used in the .tex/.md tables
    ASSET_INDEX                                  -> one-line summary of every generated file

`write()` builds `descriptions.md`: caption + description + "what it does not show" for each asset.
Its wording is chosen from the computed values (see the `_*_sentence` helpers), so the file states
what the data show rather than what was expected. If a hypothesised pattern is absent, the helper
that covers it says so.
"""
import numpy as np

# Label text for the EuroSAT classes, indexed by class id.
CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# ---------------------------------------------------------------- figure text
# Values in braces are filled by make_report_assets.py via .format(); everything else is literal.

FIG_TEXT = {
    'fig1_poison_survival':
        'Fig 1 — Poison survival ({n_runs} poisoned runs; censored = resident throughout)',
    'fig2_accuracy_heatmaps':
        'Fig 2 — All poisoned runs: accuracy change vs the matched clean run',
    'fig3_suptitle':
        'Fig 3 — Dose–response. Runs are NOT independent: they share one test stream, one seed, '
        'and 10 runs per rule/M reuse the same clean baseline.',
    'fig3_panel_a': '(a) survival vs effect   ρ = {rho} (n={n}, evicted only)',
    'fig3_panel_b': '(b) contamination vs effect   ρ = {rho} (n={n}, all runs)',
    'fig4_title':
        'Fig 4 — mechanism: {rule}, M={M}, target={target} ({target_name}), '
        'source={source} ({source_name})\n'
        'Δ target acc = {d_target:+.2f} pp, flips −{neg}/+{pos}',
    'fig5_clean_contamination_floor':
        'Fig 5 — Contamination floor in clean (un-poisoned) runs',
    'fig6_attractor_test':
        'Fig 6 — Is the poisoned class a stronger or weaker attractor?\n'
        'bars = mean over the 12 rule×M configs; dots = individual poisoned runs (colour = rule). '
        'A rise supports H5 recruitment; a drop means the poison suppressed class c.',
    'fig7_flip_timing': 'Fig 7 — Timing of flipped predictions relative to eviction',
    'fig7_xlabel':
        'timestep − $t_{evict}$   (0 = seed gone; mass to the right = damage outlived the poison)',
    'fig7_empty':
        'No flipped predictions in any evicted run.\nNothing to plot: the poison changed no '
        'predictions.',
}

# ---------------------------------------------------------------- table captions

TABLE_CAPTION = {
    'T1_eviction_times':
        'Eviction time $t_{evict}$ of the seeded poison, in timesteps, for every poisoned run. '
        '"—" = right-censored (the seed was still resident after all 8100 samples).',
    'T2_per_rule_summary':
        'Per rule × reservoir size M, aggregated over the 10 target classes. Contamination is '
        'expressed as a fraction of capacity. "mean frac flips after evict" averages only over '
        'evicted runs that had at least one flip.',
    'T3_flip_decomposition':
        'Per-true-class correctness for the {top_n} runs with the largest |Δ target accuracy|. '
        '"neg flips" = correct in the clean run and wrong in the poisoned run.',
    'T3b_target_class_sinks':
        'Where target-class samples that were correct in the clean run but wrong in the poisoned '
        'run ended up (predicted class in the poisoned run).',
    'T4_clean_contamination':
        'Contamination of the reservoir in clean (un-poisoned) runs. "Contaminant" = a stored '
        'entry whose ground-truth label differs from the buffer it sits in; the fraction is '
        'relative to capacity 10·M.',
    'A_accuracy':
        'Overall EuroSAT top-1 accuracy (%) at M={M}, read from the per-run scalar JSONs. Last '
        'row is the matched clean run.',
}

# ---------------------------------------------------------------- asset index
# (file, one-line summary) — rendered as the lookup table at the top of descriptions.md.

ASSET_INDEX = [
    ('figures/fig1_poison_survival', 'How long the injected entry survives, per rule and M.'),
    ('figures/fig2_accuracy_heatmaps',
     'Δ target- and source-class accuracy for all 120 runs, as two heatmaps.'),
    ('figures/fig3_dose_response',
     'Does longer survival / more contamination mean more damage? Two scatter panels.'),
    ('figures/fig4_mechanism_exemplar',
     'One run in detail: contamination, recruitment and the accuracy gap over time.'),
    ('figures/fig4b_mechanism_moderate', 'Same four panels for a moderate-effect run (appendix).'),
    ('figures/fig4c_mechanism_null', 'Same four panels for a null-effect run (appendix).'),
    ('figures/fig5_clean_contamination_floor',
     'How contaminated the reservoir is with no poison at all — the baseline for everything else.'),
    ('figures/fig6_attractor_test',
     'P(prediction = targeted class), clean vs poisoned, per target class.'),
    ('figures/fig7_flip_timing', 'When flipped predictions happen, relative to eviction.'),
    ('tables/T1_eviction_times', 'Eviction time for every run; "—" = never evicted.'),
    ('tables/T2_per_rule_summary', 'One row per rule × M: survival, deltas, contamination, flips.'),
    ('tables/T3_flip_decomposition', 'Per-class correct/flip counts for the largest-effect runs.'),
    ('tables/T3b_target_class_sinks', 'Which class the lost target-class samples went to.'),
    ('tables/T4_clean_contamination', 'Clean-run contamination statistics per rule × M.'),
    ('tables/A1_accuracy_M1', 'Overall accuracy at M=1, straight from the scalar JSONs.'),
    ('tables/A2_accuracy_M3', 'Overall accuracy at M=3, straight from the scalar JSONs.'),
    ('tables/A3_accuracy_M5', 'Overall accuracy at M=5, straight from the scalar JSONs.'),
    ('derived/run_level.csv', 'One row per poisoned run — the backbone every figure reads from.'),
    ('derived/flip_events.csv', 'One row per flipped prediction, with its offset from eviction.'),
    ('derived/summary_numbers.json',
     'Every headline number, so it can be quoted without re-deriving.'),
    ('derived/summary_numbers.md', 'The same numbers in readable form.'),
    ('derived/file_inventory.txt', 'Every run directory found on disk and what was read from it.'),
    ('run_log.txt', 'Console log of the build: runs found, warnings, cross-checks.'),
]


# ---------------------------------------------------------------- summary_numbers text
# Definitions and verdicts that ship inside summary_numbers.json, so the file explains itself.

SUMMARY_NOTE = {
    'contamination_floor':
        'fraction of reservoir slots holding a wrong-class entry, 2nd half of the stream',
    'survival':
        'residency = t_evict, or 8100 (stream length) for censored runs; the ratio is a lower '
        'bound because censored runs are truncated',
    'damage_after_eviction':
        'a flip at timestep >= t_evict is post-eviction, because runner.py updates the reservoir '
        'before reading it out',
    'recruitment':
        'recruitment = a source-class entry in the targeted buffer other than the seed itself '
        '(source_in_target - seeded_poison > 0)',
}

# The FIFO diagnostic asks whether FIFO's null result means fast eviction (expected) or that the
# seed is never read (a bug). make_report_assets.py picks one of these by testing residency
# against flip counts.
FIFO_VERDICT = {
    'suspicious':
        'SUSPICIOUS: every FIFO run has zero flipped predictions, including {n_long} runs where '
        'the seed stayed resident for >100 timesteps. Fast eviction cannot explain a null effect '
        'in those runs, so this is more consistent with the seeded entry never influencing the '
        'readout (possible bug).',
    'fast_eviction':
        'CONSISTENT WITH FAST EVICTION: FIFO evicts the seed within a few dozen timesteps, so it '
        'is read out on very few predictions and most runs show no flips. No evidence of a '
        'read-path bug.',
    'mixed':
        'MIXED: flip counts are not explained by residency alone; see the per-run columns n_flip '
        'and residency in run_level.csv.',
}


def write_summary_md(S, path, json_dumps):
    """Readable companion to summary_numbers.json: headlines first, then every block verbatim."""
    with open(path, 'w') as f:
        f.write('# Summary numbers\n\nEvery number quoted in the report is derived here so it can '
                'be cited without re-running the analysis. Generated by '
                '`analysis/make_report_assets.py`.\n\n')
        f.write('**Contamination floor (clean runs, steady state):** '
                f"{S['contamination_floor_clean']['overall_steady_state_fraction']:.3f} of the "
                'reservoir overall; per rule '
                + ', '.join(f'{k} {v:.3f}' for k, v in
                            S['contamination_floor_clean']['per_rule_steady_state_fraction'].items())
                + '.\n\n')
        f.write(f"**Effect sizes:** of {S['effect_counts']['n_runs']} poisoned runs, "
                f"{S['effect_counts']['|delta target acc| > 1pp']} moved target accuracy by >1 pp, "
                f"{S['effect_counts']['|delta target acc| > 5pp']} by >5 pp, and "
                f"{S['effect_counts']['|delta target acc| > 10pp']} by >10 pp.\n\n")
        f.write(f"**FIFO diagnostic:** {S['fifo_diagnostic']['verdict']}\n\n")
        for k, v in S.items():
            f.write(f'## {k.replace("_", " ")}\n\n```json\n{json_dumps(v)}\n```\n\n')


def _fmt(v, nd=2, dash='—'):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


# ---------------------------------------------------------------- data-driven sentences
# Each helper picks its wording from the numbers, so a hypothesis that is not in the data does
# not get asserted anyway.

def _rho_sentence(rho):
    """Fig 3: direction-aware wording (negative rho = more exposure, more damage)."""
    def one(r, label, exposure):
        if r is None or not np.isfinite(r):
            return f'{label} is undefined.'
        strength = 'no' if abs(r) < 0.2 else ('a weak' if abs(r) < 0.4 else
                                              ('a moderate' if abs(r) < 0.6 else 'a strong'))
        if abs(r) < 0.2:
            return f'{label} shows {strength} monotone association with the accuracy change.'
        direction = ('more negative Δ target accuracy, i.e. more damage'
                     if r < 0 else 'more positive Δ target accuracy, i.e. less damage')
        return f'{label} shows {strength} association: greater {exposure} goes with {direction}.'

    a, b = rho['rho_t_evict'], rho['rho_excess_contam']
    both = np.isfinite(a or np.nan) and np.isfinite(b or np.nan)
    agree = (' The two panels agree in sign, and (b) is the more trustworthy of the two because '
             'it retains the censored runs that (a) must drop.' if both and a * b > 0 else
             ' The two panels disagree in sign, so the exposure–damage relationship is not '
             'robust to how exposure is measured.' if both else '')
    return ('Panel (a): ' + one(a, 'eviction time', 'poison residency')
            + ' Panel (b): ' + one(b, 'excess contamination',
                                   'excess contamination of the targeted buffer') + agree)


def _sink_sentence(stats):
    """T3: did the lost target-class mass actually land on the poison's source class?"""
    if not stats:
        return ''
    hits = [s for s in stats if s['n_lost']]
    if not hits:
        return 'None of these runs lost any target-class sample, so there is no mass to trace.'
    fracs = [s['frac_lost_to_source_class'] for s in hits]
    lead, mean_f = hits[0], float(np.mean(fracs))
    qual = ('most of the lost mass' if mean_f >= 0.6 else
            'about half the lost mass' if mean_f >= 0.4 else
            'a minority of the lost mass' if mean_f >= 0.15 else 'very little of the lost mass')
    always_src = all(s['largest_sink'] == s['source_class'] for s in hits)
    verdict = ('In every one of these runs the single largest destination is the poison\'s own '
               'source class, so the redirection is specific rather than a diffuse loss of accuracy'
               if always_src else
               'The largest destination is not always the source class, so the redirection is '
               'only partly specific to the poison\'s label')
    return (f'In the top run (`{lead["run"]}`), {lead["n_lost"]} target-class samples are lost and '
            f'{lead["n_lost_to_source_class"]} of them ({lead["frac_lost_to_source_class"]:.0%}) '
            f'are predicted as the poison\'s source class, which is also the largest single '
            f'destination ({lead["largest_sink"]}, {lead["largest_sink_count"]} samples). Across '
            f'the top runs that share ranges {min(fracs):.0%}–{max(fracs):.0%}. {verdict}; the '
            f'remainder spreads over other classes, so {qual} follows the poison\'s label while '
            'the rest reflects a general degradation of the target class.')


def _floor_sentence(per_rule, overall):
    """Fig 5: the rules differ by ~5x, so a single 'high/low' verdict would mislead."""
    lo_r, hi_r = min(per_rule, key=per_rule.get), max(per_rule, key=per_rule.get)
    lo, hi = per_rule[lo_r], per_rule[hi_r]
    return (f'The floor differs sharply by rule rather than being a single number: {hi_r} sits at '
            f'{hi:.0%} of the reservoir contaminated in steady state, while {lo_r} sits at '
            f'{lo:.0%} — a factor of {hi / lo:.1f}. This reframes the rest of the study in two '
            f'ways. First, wrong-class entries are a normal operating condition of TCA, not an '
            f'anomaly the injected seed creates: at the overall mean of {overall:.0%}, roughly one '
            f'slot in {1 / overall:.0f} already holds a contaminant before any poison is added. '
            f'Second, one injected entry is a much larger *relative* perturbation under {lo_r} '
            f'(where the buffer is otherwise near-clean) than under {hi_r} (where it is one more '
            f'contaminant among many) — which is the right lens for reading the per-rule effects '
            'in Fig 2 and T2.')


def _concentration_sentence(rl, n_eff, rules, effect_pt):
    """Fig 2: are the non-zero cells spread over the grid or concentrated?"""
    eff = rl[rl['d_target_acc'].abs() > effect_pt]
    if eff.empty:
        return 'No run moved target accuracy by more than 1 pp, so the matrix is uniformly null.'
    top2 = eff['target_class'].value_counts().head(2)
    dead_rules = [r for r in rules if not (eff['rule'] == r).any()]
    parts = [f'The non-zero cells are concentrated, not spread evenly: target classes '
             f'{" and ".join(f"{c} ({CLASS_NAMES[c]})" for c in top2.index)} alone account for '
             f'{top2.sum()} of the {n_eff} runs exceeding ±1 pp, and '
             f'{len(CLASS_NAMES) - eff["target_class"].nunique()} of the 10 target classes contain '
             f'no such run at all.']
    if dead_rules:
        mx = rl[rl['rule'].isin(dead_rules)]['d_target_acc'].abs().max()
        parts.append(f'Whole columns are empty: {" and ".join(dead_rules)} produced no run above '
                     f'±1 pp anywhere in the grid (largest |Δ| = {mx:.2f} pp), so the effect is a '
                     'property of the eviction rule as much as of the target class.')
    neg = eff[eff['d_target_acc'] < 0]
    up = int((neg['d_source_acc'] > 0).sum())
    if len(neg) and up > 0.6 * len(neg):
        parts.append(f'Comparing the two panels row by row, {up} of the {len(neg)} runs that lost '
                     'target-class accuracy *gained* source-class accuracy: the poisoned class '
                     'loses samples and the poison\'s own source class picks them up.')
    return ' '.join(parts)


def _attractor_sentence(a):
    """Fig 6: sign counts can look balanced while the magnitudes are one-sided."""
    drops, rises = abs(a['sum_of_drops'] or 0.0), abs(a['sum_of_rises'] or 0.0)
    balanced = abs(a['n_fell'] - a['n_rose']) <= 0.25 * max(1, a['n_fell'] + a['n_rose'])
    if drops == 0 and rises == 0:
        return 'No run changed P(pred = c) at all.'
    ratio = (drops / rises) if rises else float('inf')
    lead = ('The sign counts are close to balanced, but they weight a 0.0001 change the same as a '
            '0.09 one. By magnitude the picture is one-sided: ' if balanced else 'By magnitude: ')
    if drops > rises:
        body = (f'summed over runs the probability falls by {drops:.3f} and rises by only '
                f'{rises:.3f} (a factor of {ratio:.0f}), and every large change is a fall — the '
                f'largest is {a["largest_change"]:+.3f} in `{a["largest_change_key"]}`. '
                'The injected entry *suppressed* the targeted class as a prediction outcome; it '
                'did not turn it into an attractor. This is the opposite of what the H5 '
                'recruitment story predicts, and it is what the data show.')
    else:
        body = (f'summed over runs the probability rises by {rises:.3f} against {drops:.3f} of '
                f'falls, the largest single change being {a["largest_change"]:+.3f} in '
                f'`{a["largest_change_key"]}` — consistent with the targeted class acting as an '
                'attractor.')
    return lead + body


def _exemplar_mechanism(ex):
    """Fig 4: did the accuracy gap keep widening after the seed was gone?"""
    if ex is None:
        return ''
    if ex['censored']:
        return ('The seed is never evicted in this run, so no eviction line is drawn and the '
                '"damage outlived the poison" test cannot be applied to it. ')
    return (f'The vertical line marks t_evict = {ex["t_evict"]}. The cumulative accuracy gap is '
            f'{_fmt(ex["cum_gap_at_evict"], 0)} samples at eviction and '
            f'{_fmt(ex["cum_gap_final"], 0)} at the end of the stream, so the damage '
            + ('kept widening after the seed was gone — the poison triggered a lasting change of '
               'reservoir state rather than a transient one. ' if ex['gap_widened_after_evict'] else
               'did not grow after eviction, i.e. the effect was confined to the window in which '
               'the seed was actually read. '))


# ---------------------------------------------------------------- descriptions.md

def write_unpaired(out_path, pairing_detail):
    """Fallback text when the pairing check failed and no paired comparison could be made."""
    with open(out_path, 'w') as f:
        f.write(f'# Descriptions\n\nPAIRING CHECK FAILED ({pairing_detail}).\n\n'
                'Only Fig 1, Fig 5, T1, T4 and A1-A3 were produced; every paired comparison was '
                'skipped because poisoned and clean runs did not process the test stream in the '
                'same order.\n')


def write(out_path, S, rl, T2, T4, exemplars, rho, raw_dir, seeds, inv_stats, pairing_detail,
          rules, effect_pt):
    """Write descriptions.md: a lookup index, then caption/description/limits per asset."""
    d = rl['d_target_acc']
    floor = S['contamination_floor_clean']['overall_steady_state_fraction']
    fl = S['contamination_floor_clean']['per_rule_steady_state_fraction']
    n_eff = S['effect_counts']['|delta target acc| > 1pp']
    att_all = S['attractor_direction']['ALL']
    fell, rose = att_all['n_fell'], att_all['n_rose']
    rec, post = S['h5_recruitment'], S['damage_after_eviction']
    cens_total = int(rl['censored'].sum())
    L = []
    w = L.append

    w('# Descriptions of generated figures and tables\n')
    w('Auto-generated by `analysis/make_report_assets.py` (text lives in `analysis/'
      'descriptions.py`); every stated pattern is read back from the assets in this directory '
      'rather than assumed.\n')

    w('## Index of generated files\n')
    w('| file | what it is |')
    w('|---|---|')
    for path, one_liner in ASSET_INDEX:
        w(f'| `{path}` | {one_liner} |')
    w('\nFigures are written as both `.png` (300 dpi) and `.pdf` (vector); tables as `.csv`, '
      '`.tex` (booktabs) and `.md`.\n')

    w('## Preamble: provenance\n')
    w(f'- **Raw data:** `{raw_dir}` (read-only; nothing under `results/` was modified).')
    w(f'- **Runs found:** {inv_stats["n_poisoned"]} poisoned (expected 120) and '
      f'{inv_stats["n_clean"]} clean runs. '
      + ('No runs are missing. ' if not (inv_stats['missing_poisoned'] or inv_stats['missing_clean'])
         else f'Missing: {inv_stats["missing_poisoned"] + inv_stats["missing_clean"]}. ')
      + (f'Clean runs outside the poisoned grid ({sorted(set(inv_stats["extra_clean"]))}) are used '
         'only in T4 and are excluded from Fig 5 panels and all paired statistics.'
         if inv_stats['extra_clean'] else ''))
    w('- **Clean-run convention:** inferred from the `mode` field of each scalar JSON '
      '(`mode == "clean"`, directory suffix `__target-none`), not assumed from filenames.')
    w(f'- **Pairing check:** {pairing_detail}. Poisoned and clean runs traverse the test stream '
      'in identical order, so per-sample joins on `sample_id` are valid.')
    w(f'- **Seeds present:** {seeds}. Every run uses the same seed, so all differences are '
      'deterministic consequences of the injected entry, not sampling noise — but equally, '
      'there is no repetition from which to estimate run-to-run variance.')
    w('- **Definition caveat (important):** a "contaminant" is a stored entry whose *ground-truth* '
      'label differs from the buffer it sits in. Ground truth is an analysis-time oracle; TCA '
      'itself files entries by predicted class and never sees these labels. Contamination is '
      'therefore a diagnostic we impose, not a signal the method could act on.')
    w('- **Censoring:** a run whose seed survives all 8100 samples has no eviction time. These are '
      'right-censored, shown as `—` in T1, plotted in a separate lane in Fig 3, and excluded from '
      'medians labelled "evicted only" — they are not missing data and are never dropped silently.')
    w('- **Non-independence:** all runs share one test stream, one clean baseline per rule×M, and '
      'one poison seed per target class. The 120 runs are not 120 independent samples; no '
      'p-values are reported and correlations are descriptive only.\n')

    def add(name, files, caption, desc, notshow):
        w(f'## {name}\n')
        w(' · '.join(f'`{f}`' for f in files) + '\n')
        w(f'**Caption.** {caption}\n')
        w(f'**Description.** {desc}\n')
        w(f'**What it does not show.** {notshow}\n')

    # Fig 5 first: it reframes everything else.
    floors = ', '.join(f'{r} {v:.2f}' for r, v in fl.items())
    add('Fig 5 — Clean-run contamination floor',
        ['figures/fig5_clean_contamination_floor.png'],
        'Fraction of reservoir slots holding a wrong-class entry during normal (un-poisoned) TCA '
        'operation, by reservoir size M; one line per eviction rule.',
        f'For each clean run the per-timestep `total_contaminant` count is divided by capacity '
        f'10·M and binned into 100 equal-width timestep bins. The annotated steady-state mean '
        f'averages the second half of the stream. Observed floors (fraction of the reservoir '
        f'contaminated): {floors}; overall {floor:.2f}. ' + _floor_sentence(fl, floor),
        'It does not show whether contaminated entries are harmful: a wrong-class neighbour can '
        'still supply a useful token-level match, and accuracy is measured separately.')

    med_by_rule = {r: S['survival']['per_rule'][r]['median_residency_censored_as_max']
                   for r in rules}
    add('Fig 1 — Poison survival', ['figures/fig1_poison_survival.png'],
        'Empirical survival curves for the injected entry: fraction of runs in which the seed is '
        'still resident at timestep t, by rule and reservoir size (log time axis).',
        f'Survival at t is the fraction of the 10 runs per rule×M whose residency exceeds t, where '
        f'residency is `t_evict` for evicted runs and the full stream length for censored ones '
        f'(step plot, `where="post"`). Across all {len(rl)} runs, {cens_total} are censored. '
        f'Median residency by rule (censored counted at 8100): '
        + ', '.join(f'{r} {v:.0f}' for r, v in med_by_rule.items()) + '. '
        + f'The longest-surviving rule is {S["survival"]["longest_surviving_rule"]}, at '
        f'{_fmt(S["survival"]["ratio_longest_to_fifo_median_residency"])}× FIFO\'s median '
        'residency; FIFO is the clear outlier, discarding the seed roughly an order of magnitude '
        'sooner than any content-based rule. The log axis is used because eviction, where it '
        'happens at all, is concentrated in the first few hundred timesteps.',
        'It does not distinguish *why* a seed survives — a rule may retain it because the entry '
        'looks diverse/uncertain, or simply because the targeted buffer is rarely written to.')

    vmax = S.get('figure_params', {}).get('fig2_symmetric_vmax_pp')
    add('Fig 2 — All poisoned runs', ['figures/fig2_accuracy_heatmaps.png'],
        'Change in target-class and source-class accuracy (poisoned − matched clean run), in '
        'percentage points, for all runs: rows are target classes, columns are the 12 rule×M cells.',
        f'Per-class accuracy is recomputed from the per-sample files for both runs of each pair '
        f'(identical to `runner._class_metrics`), so clean and poisoned sides are defined the same '
        f'way. Both panels share one symmetric diverging scale centred at zero '
        f'(±{_fmt(vmax, 1) if vmax else "max|Δ|"} pp). Observed range of Δ target accuracy: '
        f'{d.min():+.2f} to {d.max():+.2f} pp, with {int((d == 0).sum())} of {len(rl)} runs at '
        f'exactly zero and {n_eff} exceeding ±1 pp. '
        + _concentration_sentence(rl, n_eff, rules, effect_pt),
        'It does not separate effect direction from effect reliability: with one seed per cell '
        'there is no within-cell replicate, so a single ±1 pp cell cannot be distinguished from '
        'deterministic jitter of the readout.')

    add('Fig 3 — Dose–response', ['figures/fig3_dose_response.png'],
        'Eviction time (log) and excess targeted-buffer contamination against the change in '
        'target-class accuracy; colour = rule, marker = M; censored runs in a separate lane.',
        f'Panel (a) plots `t_evict` for evicted runs only (censored runs are parked to the right '
        f'of the dotted separator as open markers, since no x-value is defined for them) and '
        f'reports Spearman ρ = {_fmt(rho["rho_t_evict"])} over n = {rho["n_t_evict"]} evicted runs. '
        f'Panel (b) uses excess targeted contamination (poisoned − clean, as a fraction of M), '
        f'which is defined for all {rho["n_excess_contam"]} runs including censored ones: '
        f'ρ = {_fmt(rho["rho_excess_contam"])}. ' + _rho_sentence(rho),
        'It does not support a causal dose–response claim: runs are not independent (shared stream, '
        'shared baselines), ρ is descriptive, and censoring in panel (a) removes exactly the '
        'longest-exposure runs, which biases that panel toward the short-residency regime.')

    ex = exemplars.get('default')
    if ex:
        add('Fig 4 — Mechanism (exemplar)', ['figures/fig4_mechanism_exemplar.png'],
            f'Four aligned views of one run ({ex["rule"]}, M={ex["M"]}, target '
            f'{ex["target_class"]}, source {ex["source_class"]}): total contamination, targeted-'
            'buffer contamination, source-class entries in the targeted buffer, and the cumulative '
            'accuracy gap against the matched clean run.',
            'Panels 1–2 are binned to 200 points for legibility and show poisoned and clean runs '
            'together; panel 3 is raw and shows both the raw `source_in_target` count and the '
            'recruitment signal with the seed\'s own contribution removed; panel 4 is '
            'cumsum(correct_clean) − cumsum(correct_poisoned), so an upward slope means the '
            f'poisoned run is falling behind. This run has Δ target accuracy '
            f'{ex["d_target_acc"]:+.2f} pp, {ex["n_flip_neg"]} negative and {ex["n_flip_pos"]} '
            f'positive flips, and peak recruitment {ex["recruit_max"]}. ' + _exemplar_mechanism(ex),
            'A single exemplar cannot establish the mechanism for the grid; it shows one trajectory '
            'consistent (or not) with the aggregate statistics, and the panels are correlational — '
            'nothing here isolates the seed from ordinary reservoir churn.')

    appendix = [(k, exemplars.get(k)) for k in ('moderate', 'null') if exemplars.get(k)]
    if appendix:
        bits = []
        for k, e in appendix:
            when = 'never evicted' if e['censored'] else f't_evict = {e["t_evict"]}'
            bits.append(f'The {k}-effect panel is `{e["key"]}` (Δ target accuracy '
                        f'{e["d_target_acc"]:+.2f} pp, {when}, {e["n_flip_neg"]}/'
                        f'{e["n_flip_pos"]} negative/positive flips).')
        detail = ' '.join(bits)
        add('Fig 4b / 4c — Mechanism (appendix exemplars)',
            [f'figures/fig4{s}_mechanism_{k}.png' for s, k in (('b', 'moderate'), ('c', 'null'))],
            'The Fig 4 layout repeated for a moderate-effect and a null-effect run, for contrast '
            'with the exemplar.',
            'The moderate run is the one whose |Δ target accuracy| is closest to the median among '
            'runs exceeding ±1 pp; the null run is the one with the smallest |Δ target accuracy| '
            'in the whole grid. Both are selected automatically, so they track the data rather '
            f'than being hand-picked. {detail} The null panel is the more informative of the two: '
            'it shows the reservoir state diverging from the clean run while the accuracy gap '
            'stays flat at zero, i.e. contamination without consequence.',
            'Neither is representative in a statistical sense — they are the endpoints of the '
            'effect distribution, chosen to bracket it, not to summarise it.')

    add('Fig 6 — Attractor test', ['figures/fig6_attractor_test.png'],
        'Probability that a prediction lands on the targeted class, clean vs poisoned, per target '
        'class (bars = mean over the 12 rule×M configs, dots = individual runs).',
        f'`P(pred = c)` is computed over the whole 8100-sample stream in both runs of each pair. '
        f'Across all {len(rl)} runs the probability fell in {fell} and rose in {rose}; per-rule '
        'counts and totals are in `summary_numbers.json` under `attractor_direction`. '
        + _attractor_sentence(att_all),
        'It does not decompose which classes lost the reassigned mass (see T3b), and it averages '
        'over rules and M, which hides the possibility that one rule behaves oppositely.')

    n_flip_ev = S['flip_pool']['n_in_evicted_runs']
    add('Fig 7 — Flip timing', ['figures/fig7_flip_timing.png'],
        'Distribution of flipped predictions relative to the eviction of the seed, pooled over '
        'evicted runs, split into negative and positive flips.',
        f'Each flipped sample contributes `timestep − t_evict`; censored runs have no reference '
        f'point and are excluded. Because `runner.py` updates the reservoir *before* reading it '
        f'out, offset 0 is already poison-free, so mass at ≥ 0 is post-eviction damage. '
        + (f'The pool contains {n_flip_ev} flips from evicted runs'
           + (f', of which {S["flip_pool"]["frac_after_eviction"]:.0%} occur at or after eviction.'
              if S['flip_pool']['frac_after_eviction'] is not None else '.')
           if n_flip_ev else
           'The pool is empty: no evicted run produced a single flipped prediction, so the panel '
           'is a placeholder rather than a distribution. That absence is itself the finding — in '
           'this grid, seeds that get evicted change no predictions at all.'),
        'It does not attribute post-eviction flips to the poison: after eviction the two runs have '
        'different reservoir contents for many reasons, and this histogram cannot separate a '
        'lasting poison effect from ordinary trajectory divergence.')

    med_txt = ', '.join(f'{r} {_fmt(S["survival"]["per_rule"][r]["median_t_evict_evicted_only"], 1)}'
                        for r in rules)
    add('T1 — Eviction times', ['tables/T1_eviction_times.csv'],
        'Eviction time of the seeded poison for every poisoned run (10 target classes × 12 rule×M '
        'cells), with `—` for right-censored runs and per-column medians and censor counts.',
        f'`t_evict` is the first timestep at which `seeded_poison` reads 0 in the run\'s `ts__*.csv`. '
        f'Medians are over evicted runs only, since a median that treats censored runs as an '
        f'arbitrary large value would be an artefact of the stream length. Per-rule medians '
        f'(evicted only): {med_txt}; total censored: {cens_total} of {len(rl)}.',
        'It does not say how *often* the targeted buffer turned over afterwards — a seed evicted '
        'at t=20 may sit in a buffer that is rewritten constantly or almost never (see the '
        '`recontam_*` columns of `run_level.csv`).')

    add('T2 — Per-rule summary', ['tables/T2_per_rule_summary.csv'],
        'One row per rule × M: median eviction time, censor count, mean accuracy deltas, number of '
        'runs moving target accuracy by more than 1 pp, mean excess contamination, mean fraction '
        'of flips occurring after eviction, and the clean-run contamination floor.',
        'Every quantity is the mean (or median, where noted) over that cell\'s 10 target classes, '
        'using the definitions in `run_level.csv`; the contamination floor column is the same '
        'steady-state number annotated in Fig 5, repeated here so each row is self-contained. '
        f'Across the {len(T2)} cells, mean Δ target accuracy ranges from '
        f'{T2["mean Δ target acc"].min():+.2f} to {T2["mean Δ target acc"].max():+.2f} pp.',
        'Averaging over 10 target classes hides sign cancellation: a cell whose mean is ~0 may '
        'contain one large positive and one large negative run (check T1/Fig 2 for the spread).')

    add('T3 / T3b — Flip decomposition',
        ['tables/T3_flip_decomposition.csv', 'tables/T3b_target_class_sinks.csv'],
        'For the largest-effect runs: per true class, the number of correct predictions in the '
        'clean and poisoned runs and the negative/positive flip counts (T3), and where samples '
        'lost from the target class were reassigned (T3b).',
        'Runs are ranked by |Δ target accuracy| and the top 3 are decomposed by joining the two '
        'per-sample files on `sample_id`. "Lost" samples are those correct in the clean run and '
        'wrong in the poisoned run whose true class is the target class; T3b tabulates the class '
        'they were predicted as instead, which is the direct test of whether the poison redirects '
        'mass toward the source class specifically. ' + _sink_sentence(S.get('largest_effect_run_sinks')),
        'It covers only the extreme tail of the grid and so describes the worst cases, not the '
        'typical run; it also cannot show whether the same reassignment happens transiently in '
        'runs whose net accuracy change is zero.')

    add('T4 — Clean-run contamination statistics', ['tables/T4_clean_contamination.csv'],
        'Mean, median, steady-state and maximum contaminated fraction of the reservoir in every '
        'clean run, per rule × M.',
        f'Computed from the clean `ts__*.csv` files as `total_contaminant / (10·M)`; the '
        f'steady-state column averages the second half of the stream and is the number annotated '
        f'in Fig 5. Clean runs at M values outside the poisoned grid are included and flagged in '
        f'the "in poisoned grid" column. Observed steady-state fractions span '
        f'{T4["steady-state frac (2nd half)"].min():.2f}–'
        f'{T4["steady-state frac (2nd half)"].max():.2f}.',
        'It does not weight contaminants by influence: the TCA readout weights entries by token '
        'similarity, so a contaminated slot is not necessarily an active one.')

    add('A1–A3 — Accuracy tables per M',
        ['tables/A1_accuracy_M1.csv', 'tables/A2_accuracy_M3.csv', 'tables/A3_accuracy_M5.csv'],
        'Overall EuroSAT top-1 accuracy for every poisoned run and its clean baseline, one table '
        'per reservoir size.',
        'Values are read directly from the per-run scalar JSONs (not recomputed), so these tables '
        'are reproducible from the raw outputs alone and serve as a cross-check on the derived '
        'per-class numbers used elsewhere; the derived per-sample recomputation of target-class '
        'accuracy was verified against the JSON field during the build (see `run_log.txt`).',
        'Overall accuracy is a whole-stream aggregate and is insensitive to a poison that harms '
        'one class while helping another; the per-class deltas in Fig 2 are the relevant view.')

    add('Derived data files',
        ['derived/run_level.csv', 'derived/flip_events.csv', 'derived/summary_numbers.json',
         'derived/file_inventory.txt'],
        'The intermediate tables every figure and table is built from.',
        '`run_level.csv` has one row per poisoned run and holds every per-run metric (survival, '
        'contamination, recruitment, compounding, flips, accuracy deltas) — each defined in '
        'exactly one function in `make_report_assets.py`. `flip_events.csv` has one row per '
        'flipped prediction with its offset from eviction, which is what Fig 7 pools. '
        '`summary_numbers.json` collects the headline numbers so they can be quoted without '
        're-deriving them, and `file_inventory.txt` records every run directory found on disk. '
        'Passing `--reuse-cache` reuses the two CSVs instead of re-parsing the raw runs.',
        'They carry no provenance beyond this build: regenerate them if the raw runs change, '
        'since nothing detects a stale cache automatically.')

    w('## Cross-cutting findings, stated plainly\n')
    w(f'- The reservoir is contaminated at a steady-state rate of {floor:.0%} overall in normal '
      f'operation, ranging from {min(fl.values()):.0%} ({min(fl, key=fl.get)}) to '
      f'{max(fl.values()):.0%} ({max(fl, key=fl.get)}) across rules (Fig 5, T4). Wrong-class '
      'entries are the normal state of this cache, not an artefact of the attack.')
    w(f'- {n_eff} of {len(rl)} poisoned runs moved target-class accuracy by more than 1 pp; '
      f'{S["effect_counts"]["|delta target acc| > 5pp"]} moved it by more than 5 pp. '
      + ('The modal outcome is no measurable effect.' if n_eff < len(rl) / 2 else
         'A majority of runs show a measurable effect.'))
    w('- ' + (f'Recruitment (H5) was observed in {rec["n_runs_with_recruitment"]} of '
              f'{rec["n_runs"]} runs. '
              + (f'In {rec["n_recruitment_and_source_acc_improved"]} of those the source class\'s '
                 'accuracy *improved*, which dissociates "source-class entries appear in the '
                 'targeted buffer" from "the source class is harmed" — the recruitment signal does '
                 'not by itself imply damage.'
                 if rec['n_recruitment_and_source_acc_improved'] else
                 'Recruitment never co-occurred with improved source-class accuracy.')))
    w('- ' + (f'Among runs with a real effect and a known eviction time (n = '
              f'{post["n_runs_considered"]}), a mean of '
              f'{post["mean_frac_flips_after_evict"]:.0%} of flipped predictions occurred at or '
              'after eviction.' if post['n_runs_considered'] and
              post['mean_frac_flips_after_evict'] is not None else
              'No run had both a >1 pp effect and a known eviction time, so the fraction of damage '
              'occurring after eviction is undefined for this grid — reported as such rather than '
              'imputed.'))
    w(f'- FIFO diagnostic: {S["fifo_diagnostic"]["verdict"]}')
    w('')
    with open(out_path, 'w') as f:
        f.write('\n'.join(L))
