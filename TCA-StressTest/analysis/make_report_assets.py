#!/usr/bin/env python
"""Build tables, figures and summary numbers for the TCA reservoir-contamination study.

Reads the raw per-run outputs written by runner.py (results/runs/<key>/{ts,persample,scalar}*)
and writes every derived asset under results-inferred/.  Nothing under results/ is modified.

This module computes and draws; it holds no report prose. Every caption, figure title and the
whole of descriptions.md live in descriptions.py, keyed by output filename.

Usage:
    python analysis/make_report_assets.py                    # full build
    python analysis/make_report_assets.py --dry-run          # inventory + pairing check only
    python analysis/make_report_assets.py --reuse-cache      # skip re-parsing ~2M raw rows
"""
import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import descriptions
from descriptions import CLASS_NAMES, FIG_TEXT, TABLE_CAPTION

# ---------------------------------------------------------------- constants

NUM_CLASSES = len(CLASS_NAMES)
RULES = ['fifo', 'similarity', 'diversity', 'uncertainty']
M_VALUES = [1, 3, 5]                       # M values that have poisoned runs
# Okabe-Ito, colourblind-safe; one fixed colour per rule everywhere.
RULE_COLOR = {'fifo': '#0072B2', 'similarity': '#D55E00',
              'diversity': '#009E73', 'uncertainty': '#CC79A7'}
M_MARKER = {1: 'o', 3: 's', 5: '^'}
# diversity and uncertainty behave almost identically, so line plots need a second channel
RULE_STYLE = {'fifo': '-', 'similarity': '--', 'diversity': '-.', 'uncertainty': ':'}
EFFECT_PT = 1.0                            # |delta target acc| above which a run "has an effect"

plt.rcParams.update({'figure.dpi': 110, 'savefig.bbox': 'tight', 'font.size': 9,
                     'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True})


class Log:
    """print() that also tees to run_log.txt."""

    def __init__(self, path=None):
        self.fh = open(path, 'w') if path else None

    def __call__(self, *parts):
        msg = ' '.join(str(p) for p in parts)
        print(msg)
        if self.fh:
            self.fh.write(msg + '\n')
            self.fh.flush()

    def warn(self, *parts):
        self('WARNING:', *parts)


# ---------------------------------------------------------------- discovery

def discover_runs(raw_dir, log):
    """Walk raw_dir and return an inventory DataFrame, one row per run directory found.

    The clean-run filename convention is not assumed: every directory is classified by the
    `mode` field of its scalar JSON, and the directory name is only used as a fallback.
    """
    rows = []
    for d in sorted(glob.glob(os.path.join(raw_dir, '*'))):
        if not os.path.isdir(d):
            continue
        scalars = glob.glob(os.path.join(d, 'scalar__*.json'))
        ts = glob.glob(os.path.join(d, 'ts__*.csv'))
        ps = glob.glob(os.path.join(d, 'persample__*.csv'))
        key = os.path.basename(d)
        if not scalars:
            log.warn(f'{key}: no scalar JSON, skipped')
            continue
        with open(scalars[0]) as f:
            js = json.load(f)
        rows.append({
            'key': key, 'dir': d,
            'rule': js.get('rule'), 'M': js.get('M'), 'mode': js.get('mode'),
            'target_class': js.get('target_class'), 'source_class': js.get('source_class'),
            'overall_acc': js.get('overall_acc'), 'avg_conf_gt': js.get('avg_conf_gt'),
            'target_class_acc': js.get('target_class_acc'),
            'source_class_acc': js.get('source_class_acc'),
            'seed': js.get('seed'), 'n_samples': js.get('n_samples'),
            'scalar_path': scalars[0],
            'ts_path': ts[0] if ts else None,
            'persample_path': ps[0] if ps else None,
        })
    inv = pd.DataFrame(rows)
    if inv.empty:
        log.warn(f'no runs found under {raw_dir}')
        return inv

    log(f'Found {len(inv)} run directories under {raw_dir}')
    log('  modes present: ' + ', '.join(f'{k}={v}' for k, v in inv['mode'].value_counts().items()))
    log('  clean-run naming convention (inferred from mode field): '
        + ', '.join(sorted(inv.loc[inv['mode'] == 'clean', 'key'].head(2))) + ' ...')
    for col, label in [('ts_path', 'ts__*.csv'), ('persample_path', 'persample__*.csv')]:
        missing = inv.loc[inv[col].isna(), 'key'].tolist()
        if missing:
            log.warn(f'{len(missing)} runs missing {label}: {missing[:5]}')
    return inv


def check_expected_grid(inv, log):
    """Report poisoned/clean runs found vs the 120 + 12 the design calls for."""
    pois = inv[inv['mode'] == 'seeded']
    clean = inv[inv['mode'] == 'clean']
    missing_p, missing_c, extra_c = [], [], []
    for rule in RULES:
        for M in M_VALUES:
            if not ((clean['rule'] == rule) & (clean['M'] == M)).any():
                missing_c.append(f'{rule}/M{M}')
            for c in range(NUM_CLASSES):
                hit = ((pois['rule'] == rule) & (pois['M'] == M) & (pois['target_class'] == c))
                if not hit.any():
                    missing_p.append(f'{rule}/M{M}/target-{c}')
    for _, r in clean.iterrows():
        if r['M'] not in M_VALUES:
            extra_c.append(f"{r['rule']}/M{r['M']}")
    log(f'Poisoned runs: {len(pois)}/120 expected; clean runs: {len(clean)} '
        f'({len(clean[clean["M"].isin(M_VALUES)])} at M in {M_VALUES}, expected 12)')
    if missing_p:
        log.warn(f'{len(missing_p)} poisoned runs missing: {missing_p}')
    if missing_c:
        log.warn(f'{len(missing_c)} clean runs missing: {missing_c}')
    if extra_c:
        log(f'  extra clean runs outside the poisoned grid (kept for T4/Fig5 only): {sorted(extra_c)}')
    return {'n_poisoned': len(pois), 'n_clean': len(clean),
            'missing_poisoned': missing_p, 'missing_clean': missing_c, 'extra_clean': extra_c}


def check_pairing(inv, log):
    """Verify every run walked the test stream in the same order (sample_id and true_class).

    Returns (ok, detail). If this fails, all paired (poisoned-vs-clean) analysis is skipped.
    """
    have = inv[inv['persample_path'].notna()]
    if have.empty:
        return False, 'no per-sample files'
    ref_key = have.iloc[0]['key']
    ref = pd.read_csv(have.iloc[0]['persample_path'], usecols=['sample_id', 'true_class'])
    bad = []
    for _, r in have.iterrows():
        df = pd.read_csv(r['persample_path'], usecols=['sample_id', 'true_class'])
        if len(df) != len(ref) or not df['sample_id'].equals(ref['sample_id']) \
                or not df['true_class'].equals(ref['true_class']):
            bad.append(r['key'])
    ok = not bad
    if ok:
        log(f'Pairing check PASSED: all {len(have)} runs share the sample_id/true_class '
            f'sequence of {ref_key} ({len(ref)} samples).')
    else:
        log.warn('!' * 70)
        log.warn(f'PAIRING CHECK FAILED for {len(bad)} runs: {bad[:10]}')
        log.warn('Poisoned and clean runs did NOT process the stream in the same order.')
        log.warn('ALL PAIRED COMPARISONS (flips, deltas, Figs 2,3,4,6,7) ARE SKIPPED.')
        log.warn('!' * 70)
    return ok, (f'passed ({len(have)} runs, {len(ref)} samples, reference {ref_key})' if ok
                else f'FAILED for {len(bad)} runs: {bad}')


# ---------------------------------------------------------------- metrics (each defined once)

def evict_time(ts):
    """First timestep at which the seeded poison is gone; None if it survives the stream.

    None means right-censored (the seed was still resident at the last sample), not missing.
    """
    gone = ts.index[ts['seeded_poison'] == 0]
    return int(ts['timestep'].iloc[gone[0]]) if len(gone) else None


def contamination_metrics(ts, M, target):
    """Mean contamination, normalised by capacity so M=1/3/5 are comparable."""
    tgt_col = f'c{target}_contam'
    return {
        'mean_total_contam': ts['total_contaminant'].mean(),
        'frac_total_contam': ts['total_contaminant'].mean() / (NUM_CLASSES * M),
        'mean_target_contam': ts[tgt_col].mean(),
        'frac_target_contam': ts[tgt_col].mean() / M,
    }


def recruitment_metrics(ts):
    """H5: source-class entries pulled into the targeted buffer, excluding the seed itself.

    ts['source_in_target'] counts the seed too (its true label *is* the source class), so
    genuine recruitment is source_in_target - seeded_poison.
    """
    recruit = (ts['source_in_target'] - ts['seeded_poison']).clip(lower=0)
    nz = ts.index[recruit > 0]
    return {
        'recruit_max': int(recruit.max()),
        'recruit_mean': float(recruit.mean()),
        't_first_recruit': int(ts['timestep'].iloc[nz[0]]) if len(nz) else None,
        'src_in_target_max_raw': int(ts['source_in_target'].max()),
    }


def compounding_metrics(ts, t_evict, target):
    """Does the targeted buffer get contaminated again once the seed is gone, and how often?"""
    if t_evict is None:
        return {'recontam_after_evict': None, 'recontam_episodes': None,
                'frac_time_recontam_after_evict': None}
    after = ts[ts['timestep'] >= t_evict][f'c{target}_contam'].values > 0
    if after.size == 0:
        return {'recontam_after_evict': False, 'recontam_episodes': 0,
                'frac_time_recontam_after_evict': 0.0}
    rising = int(np.sum(after[1:] & ~after[:-1]) + (1 if after[0] else 0))
    return {'recontam_after_evict': bool(after.any()), 'recontam_episodes': rising,
            'frac_time_recontam_after_evict': float(after.mean())}


def prediction_metrics(ps_p, ps_c, t_evict, target):
    """Paired per-sample comparison, joined on sample_id.

    A 'flip' is a sample whose correctness changed.  neg = clean-correct -> poisoned-wrong.
    Note runner.py updates the reservoir *before* reading it out, so the prediction at
    timestep t_evict is already poison-free: post-eviction means timestep >= t_evict.
    """
    j = ps_c.merge(ps_p, on='sample_id', suffixes=('_c', '_p'))
    neg = (j['correct_c'] == 1) & (j['correct_p'] == 0)
    pos = (j['correct_c'] == 0) & (j['correct_p'] == 1)
    flip = neg | pos
    diverge = j.index[j['pred_class_c'] != j['pred_class_p']]
    out = {
        'n_flip_neg': int(neg.sum()), 'n_flip_pos': int(pos.sum()), 'n_flip': int(flip.sum()),
        't_first_divergence': int(j['timestep_p'].iloc[diverge[0]]) if len(diverge) else None,
        'n_pred_diff': int(len(diverge)),
        'p_pred_target_clean': float((j['pred_class_c'] == target).mean()),
        'p_pred_target_poisoned': float((j['pred_class_p'] == target).mean()),
    }
    if t_evict is not None and flip.sum() > 0:
        out['frac_flips_after_evict'] = float((j.loc[flip, 'timestep_p'] >= t_evict).mean())
    else:
        out['frac_flips_after_evict'] = np.nan   # censored, or no flips to attribute
    return out


def class_accuracy(ps, cls):
    """Accuracy (%) over samples whose TRUE class is cls; mirrors runner._class_metrics."""
    rows = ps[ps['true_class'] == cls]
    return float(100.0 * rows['correct'].mean()) if len(rows) else np.nan


def flip_events(ps_p, ps_c, t_evict):
    """Per-flip rows for Fig 7: offset of each flipped sample relative to eviction."""
    j = ps_c.merge(ps_p, on='sample_id', suffixes=('_c', '_p'))
    neg = (j['correct_c'] == 1) & (j['correct_p'] == 0)
    pos = (j['correct_c'] == 0) & (j['correct_p'] == 1)
    sel = j[neg | pos].copy()
    if sel.empty:
        return sel.assign(sign=[], offset=[])[['sign', 'offset']]
    sel['sign'] = np.where(neg[neg | pos], 'negative', 'positive')
    sel['offset'] = np.nan if t_evict is None else sel['timestep_p'] - t_evict
    sel['timestep'] = sel['timestep_p']
    return sel[['timestep', 'sign', 'offset', 'true_class_c', 'pred_class_c', 'pred_class_p']]


# ---------------------------------------------------------------- run-level table

def load_ts(path):
    return pd.read_csv(path)


def load_ps(path):
    return pd.read_csv(path, usecols=['timestep', 'sample_id', 'true_class', 'pred_class',
                                      'correct', 'conf_gt'])


def build_run_level(inv, paired_ok, log):
    """Collapse every poisoned run to one row; also return pooled flip events."""
    clean = inv[inv['mode'] == 'clean']
    clean_cache = {}                                   # (rule, M) -> (ts, persample, scalar row)
    for _, r in clean.iterrows():
        clean_cache[(r['rule'], r['M'])] = (load_ts(r['ts_path']), load_ps(r['persample_path']), r)
    log(f'Loaded {len(clean_cache)} clean runs into memory.')

    rows, flips = [], []
    pois = inv[inv['mode'] == 'seeded'].sort_values(['rule', 'M', 'target_class'])
    for n, (_, r) in enumerate(pois.iterrows(), 1):
        if n % 20 == 0 or n == len(pois):
            log(f'  ...parsed {n}/{len(pois)} poisoned runs')
        M, target, source = int(r['M']), int(r['target_class']), int(r['source_class'])
        ts = load_ts(r['ts_path'])
        ps = load_ps(r['persample_path'])
        t_ev = evict_time(ts)

        row = {'key': r['key'], 'rule': r['rule'], 'M': M, 'target_class': target,
               'source_class': source, 'target_name': CLASS_NAMES[target],
               'source_name': CLASS_NAMES[source], 'seed': r['seed'],
               't_evict': t_ev, 'censored': t_ev is None,
               'residency': (len(ts) if t_ev is None else t_ev),
               'overall_acc_pois': r['overall_acc'],
               'target_acc_pois': class_accuracy(ps, target),
               'source_acc_pois': class_accuracy(ps, source),
               'target_acc_pois_json': r['target_class_acc'],
               'source_acc_pois_json': r['source_class_acc']}
        row.update(contamination_metrics(ts, M, target))
        row.update(recruitment_metrics(ts))
        row.update(compounding_metrics(ts, t_ev, target))

        pair = clean_cache.get((r['rule'], M))
        if pair is None:
            log.warn(f'{r["key"]}: no clean run for {r["rule"]}/M{M}; paired columns left blank')
        elif paired_ok:
            cts, cps, cjs = pair
            cc = contamination_metrics(cts, M, target)
            row['excess_frac_total_contam'] = row['frac_total_contam'] - cc['frac_total_contam']
            row['excess_frac_target_contam'] = row['frac_target_contam'] - cc['frac_target_contam']
            row['clean_frac_total_contam'] = cc['frac_total_contam']
            row['clean_frac_target_contam'] = cc['frac_target_contam']
            row['overall_acc_clean'] = cjs['overall_acc']
            row['target_acc_clean'] = class_accuracy(cps, target)
            row['source_acc_clean'] = class_accuracy(cps, source)
            row['d_overall_acc'] = row['overall_acc_pois'] - cjs['overall_acc']
            row['d_target_acc'] = row['target_acc_pois'] - row['target_acc_clean']
            row['d_source_acc'] = row['source_acc_pois'] - row['source_acc_clean']
            row.update(prediction_metrics(ps, cps, t_ev, target))
            fe = flip_events(ps, cps, t_ev)
            if len(fe):
                fe = fe.assign(key=r['key'], rule=r['rule'], M=M, target_class=target,
                               censored=t_ev is None)
                flips.append(fe)
        rows.append(row)

    run_level = pd.DataFrame(rows)
    flips_df = (pd.concat(flips, ignore_index=True) if flips
                else pd.DataFrame(columns=['timestep', 'sign', 'offset', 'key', 'rule', 'M',
                                           'target_class', 'censored']))
    if paired_ok and 'target_acc_pois_json' in run_level:
        d = (run_level['target_acc_pois'] - run_level['target_acc_pois_json']).abs().max()
        log(f'Cross-check: max |per-sample target acc - JSON target acc| = {d:.6f}')
    return run_level, flips_df


def clean_run_frames(inv):
    """(rule, M) -> clean timeseries, for the contamination-floor figure/table."""
    out = {}
    for _, r in inv[inv['mode'] == 'clean'].iterrows():
        if r['ts_path']:
            out[(r['rule'], int(r['M']))] = load_ts(r['ts_path'])
    return out


# ---------------------------------------------------------------- small helpers

def spearman(x, y):
    """Spearman rho via Pearson on average ranks (numpy/pandas only); nan if degenerate."""
    s = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(s) < 3 or s['x'].nunique() < 2 or s['y'].nunique() < 2:
        return np.nan, len(s)
    rx, ry = s['x'].rank(), s['y'].rank()
    return float(np.corrcoef(rx, ry)[0, 1]), len(s)


def bin_series(x, y, nbins):
    """Mean of y within nbins equal-width bins of x (for readable long time series)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    edges = np.linspace(x.min(), x.max() + 1e-9, nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    xs = np.array([x[idx == b].mean() if (idx == b).any() else np.nan for b in range(nbins)])
    ys = np.array([y[idx == b].mean() if (idx == b).any() else np.nan for b in range(nbins)])
    ok = ~np.isnan(xs)
    return xs[ok], ys[ok]


def cell_label(rule, M):
    return f'{rule[:4]}/M{M}'


def savefig(fig, out_dir, name, log):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'), dpi=300)
    plt.close(fig)
    log(f'  wrote figures/{name}.png + .pdf')


def fmt(v, nd=2, dash='—'):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


def emit_table(df, name, caption, out_dir, log, index=True, float_fmt='%.3f'):
    """Write one table as .csv, booktabs .tex and .md."""
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f'{name}.csv'), index=index)

    d = df.reset_index() if index else df.copy()
    cols = [str(c) for c in d.columns]

    def cell(v):
        if isinstance(v, float):
            return '—' if not np.isfinite(v) else (float_fmt % v)
        return '' if v is None else str(v)

    body = [[cell(v) for v in row] for row in d.itertuples(index=False)]

    def md_escape(s):                      # a bare '|' would split the cell
        return s.replace('|', r'\|')

    with open(os.path.join(out_dir, f'{name}.md'), 'w') as f:
        f.write(f'**{name}.** {caption}\n\n')
        f.write('| ' + ' | '.join(md_escape(c) for c in cols) + ' |\n')
        f.write('|' + '|'.join(['---'] * len(cols)) + '|\n')
        for row in body:
            f.write('| ' + ' | '.join(md_escape(c) for c in row) + ' |\n')

    # pdflatex without unicode packages chokes on these, so map them to math-mode equivalents
    TEX_UNICODE = {'—': r'\textemdash{}', 'Δ': r'$\Delta$', '×': r'$\times$', '·': r'$\cdot$',
                   '±': r'$\pm$', 'ρ': r'$\rho$', '≥': r'$\geq$', '−': '-', '’': "'",
                   '"': r'\textquotedbl{}', '>': '$>$', '<': '$<$'}

    def tex_escape(s):
        for ch in ('_', '%', '&', '#'):
            s = s.replace(ch, '\\' + ch)
        for u, t in TEX_UNICODE.items():
            s = s.replace(u, t)
        return s.replace('|', r'$|$')

    with open(os.path.join(out_dir, f'{name}.tex'), 'w') as f:
        f.write('\\begin{table}[t]\n\\centering\n\\small\n')
        f.write(f'\\caption{{{tex_escape(caption)}}}\n\\label{{tab:{name.lower()}}}\n')
        f.write('\\begin{tabular}{' + 'l' + 'r' * (len(cols) - 1) + '}\n\\toprule\n')
        f.write(' & '.join(tex_escape(c) for c in cols) + ' \\\\\n\\midrule\n')
        for row in body:
            f.write(' & '.join(tex_escape(c) for c in row) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    log(f'  wrote tables/{name}.{{csv,tex,md}}')


# ---------------------------------------------------------------- figures

def fig5_clean_floor(clean_ts, fig_dir, log):
    """Clean runs only: contaminated fraction of the reservoir over time. Run first."""
    fig, axes = plt.subplots(1, len(M_VALUES), figsize=(12, 3.4), sharey=True)
    steady = {}
    for ax, M in zip(axes, M_VALUES):
        for rule in RULES:
            ts = clean_ts.get((rule, M))
            if ts is None:
                continue
            frac = ts['total_contaminant'] / (NUM_CLASSES * M)
            xs, ys = bin_series(ts['timestep'], frac, 100)
            ax.plot(xs, ys, color=RULE_COLOR[rule], ls=RULE_STYLE[rule], lw=1.4, label=rule)
            steady[(rule, M)] = float(frac.iloc[len(frac) // 2:].mean())
        ax.set_title(f'M = {M}')
        ax.set_xlabel('timestep')
        ax.set_ylim(0, 1)
    axes[0].set_ylabel('contaminated fraction of reservoir')
    # annotate in the (empty) top-right so the box never covers the flat lines it describes
    for ax, M in zip(axes, M_VALUES):
        txt = '\n'.join(f'{r}: {steady[(r, M)]:.2f}' for r in RULES if (r, M) in steady)
        ax.text(0.98, 0.97, 'steady-state mean\n' + txt, transform=ax.transAxes,
                ha='right', va='top', fontsize=7,
                bbox=dict(fc='white', ec='0.7', alpha=0.9))
    axes[0].legend(loc='center left', fontsize=7, title='rule')
    fig.suptitle(FIG_TEXT['fig5_clean_contamination_floor'], y=1.03)
    savefig(fig, fig_dir, 'fig5_clean_contamination_floor', log)
    return steady


def fig1_survival(rl, fig_dir, log):
    """Empirical survival of the seed; censored runs stay resident to the end."""
    fig, axes = plt.subplots(1, len(M_VALUES), figsize=(12, 3.4), sharey=True)
    n_total = len(rl)
    grid = np.unique(np.concatenate([[0], np.logspace(0, np.log10(8100), 300)])).astype(int)
    for ax, M in zip(axes, M_VALUES):
        for rule in RULES:
            sub = rl[(rl['rule'] == rule) & (rl['M'] == M)]
            if sub.empty:
                continue
            resid = sub['residency'].values          # = t_evict, or stream length if censored
            surv = [(resid > t).mean() for t in grid]
            ax.step(np.maximum(grid, 1), surv, where='post', color=RULE_COLOR[rule],
                    ls=RULE_STYLE[rule], lw=1.6,
                    label=f'{rule} (n={len(sub)}, {int(sub["censored"].sum())} cens.)')
        ax.set_xscale('log')
        ax.set_xlim(1, 9000)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'M = {M}')
        ax.set_xlabel('timestep (log)')
        ax.legend(fontsize=6.5, loc='lower left')
    axes[0].set_ylabel('fraction of seeds still resident')
    fig.suptitle(FIG_TEXT['fig1_poison_survival'].format(n_runs=n_total), y=1.03)
    savefig(fig, fig_dir, 'fig1_poison_survival', log)


def _cells(rl):
    return [(r, M) for r in RULES for M in M_VALUES
            if ((rl['rule'] == r) & (rl['M'] == M)).any()]


def _matrix(rl, value_col):
    cells = _cells(rl)
    mat = np.full((NUM_CLASSES, len(cells)), np.nan)
    for j, (rule, M) in enumerate(cells):
        sub = rl[(rl['rule'] == rule) & (rl['M'] == M)]
        for _, r in sub.iterrows():
            mat[int(r['target_class']), j] = r[value_col]
    return mat, [cell_label(r, M) for r, M in cells]


def fig2_heatmaps(rl, fig_dir, log):
    m_t, cols = _matrix(rl, 'd_target_acc')
    m_s, _ = _matrix(rl, 'd_source_acc')
    vmax = np.nanmax(np.abs(np.concatenate([m_t.ravel(), m_s.ravel()])))
    vmax = float(np.ceil(vmax)) if np.isfinite(vmax) and vmax > 0 else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for ax, mat, title in [(axes[0], m_t, 'Δ target-class accuracy (pp)'),
                           (axes[1], m_s, 'Δ source-class accuracy (pp)')]:
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_yticklabels([f'{i} {n}' for i, n in enumerate(CLASS_NAMES)], fontsize=7)
        ax.set_title(title)
        ax.grid(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f'{mat[i, j]:.1f}', ha='center', va='center', fontsize=6,
                            color='white' if abs(mat[i, j]) > 0.6 * vmax else 'black')
        fig.colorbar(im, ax=ax, shrink=0.85, label='poisoned − clean (pp)')
    axes[0].set_ylabel('target class (poisoned buffer)')
    fig.suptitle(FIG_TEXT['fig2_accuracy_heatmaps'], y=1.0)
    savefig(fig, fig_dir, 'fig2_accuracy_heatmaps', log)
    return float(vmax)


def fig3_dose_response(rl, fig_dir, log):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ev, cens = rl[~rl['censored']], rl[rl['censored']]
    x_cens = 2.2e4                                   # parking lane for censored runs
    rng = np.random.default_rng(0)                   # fixed jitter so the figure is reproducible
    for rule in RULES:
        for M in M_VALUES:
            s = ev[(ev['rule'] == rule) & (ev['M'] == M)]
            axes[0].scatter(s['t_evict'].clip(lower=1), s['d_target_acc'],
                            c=RULE_COLOR[rule], marker=M_MARKER[M], s=34, alpha=0.85,
                            edgecolor='none')
            c = cens[(cens['rule'] == rule) & (cens['M'] == M)]
            # censored runs: open markers, so they are never mistaken for observed evictions
            axes[0].scatter(np.full(len(c), x_cens) * rng.uniform(0.85, 1.15, len(c)),
                            c['d_target_acc'], marker=M_MARKER[M], s=40,
                            facecolors='none', edgecolors=RULE_COLOR[rule], linewidths=1.3)
            s2 = rl[(rl['rule'] == rule) & (rl['M'] == M)]
            axes[1].scatter(s2['excess_frac_target_contam'], s2['d_target_acc'],
                            c=RULE_COLOR[rule], marker=M_MARKER[M], s=34, alpha=0.85,
                            edgecolor='none')
    rho_a, n_a = spearman(ev['t_evict'], ev['d_target_acc'])
    rho_b, n_b = spearman(rl['excess_frac_target_contam'], rl['d_target_acc'])

    axes[0].set_xscale('log')
    axes[0].axvline(1.2e4, color='0.5', ls=':', lw=1)
    axes[0].set_xlim(0.8, 4e4)
    axes[0].set_xticks([1, 10, 100, 1000, 8100, x_cens])
    axes[0].set_xticklabels(['1', '10', '100', '1000', '8100', 'censored'], fontsize=7)
    axes[0].set_xlabel('eviction time $t_{evict}$ (log)')
    axes[0].set_title(FIG_TEXT['fig3_panel_a'].format(rho=fmt(rho_a), n=n_a))
    axes[1].set_xlabel('excess targeted-buffer contamination (poisoned − clean, fraction of M)')
    axes[1].set_title(FIG_TEXT['fig3_panel_b'].format(rho=fmt(rho_b), n=n_b))
    for ax in axes:
        ax.axhline(0, color='0.4', lw=0.8)
        ax.set_ylabel('Δ target-class accuracy (pp)')
    handles = [plt.Line2D([], [], color=RULE_COLOR[r], marker='o', ls='', label=r) for r in RULES]
    handles += [plt.Line2D([], [], color='0.3', marker=M_MARKER[m], ls='', label=f'M={m}')
                for m in M_VALUES]
    handles += [plt.Line2D([], [], color='0.3', marker='o', ls='', mfc='none',
                           label='censored (panel a)')]
    axes[1].legend(handles=handles, fontsize=6.5, ncol=2, loc='best')
    fig.suptitle(FIG_TEXT['fig3_suptitle'], y=1.04, fontsize=8.5)
    savefig(fig, fig_dir, 'fig3_dose_response', log)
    return {'rho_t_evict': rho_a, 'n_t_evict': n_a,
            'rho_excess_contam': rho_b, 'n_excess_contam': n_b}


def fig4_mechanism(rl, inv, name, rule, M, target, fig_dir, log):
    """Four stacked panels for one exemplar run, poisoned vs its clean pair."""
    sub = rl[(rl['rule'] == rule) & (rl['M'] == M) & (rl['target_class'] == target)]
    if sub.empty:
        log.warn(f'exemplar {rule}/M{M}/target-{target} not found; skipping {name}')
        return None
    row = sub.iloc[0]
    pr = inv[inv['key'] == row['key']].iloc[0]
    cr = inv[(inv['mode'] == 'clean') & (inv['rule'] == rule) & (inv['M'] == M)]
    if cr.empty:
        log.warn(f'no clean pair for {rule}/M{M}; skipping {name}')
        return None
    cr = cr.iloc[0]
    tsp, tsc = load_ts(pr['ts_path']), load_ts(cr['ts_path'])
    psp, psc = load_ps(pr['persample_path']), load_ps(cr['persample_path'])
    t_ev = None if row['censored'] else int(row['t_evict'])

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 9), sharex=True)
    col_p, col_c = RULE_COLOR[rule], '0.55'
    for ax, key, lab in [(axes[0], 'total_contaminant', f'total contaminants (max {NUM_CLASSES*M})'),
                         (axes[1], f'c{target}_contam', f'buffer {target} contaminants (max {M})')]:
        for ts, c, l in [(tsc, col_c, 'clean'), (tsp, col_p, 'poisoned')]:
            xs, ys = bin_series(ts['timestep'], ts[key], 200)
            ax.plot(xs, ys, color=c, lw=1.3, label=l)
        ax.set_ylabel(lab, fontsize=8)
        ax.legend(fontsize=7, loc='upper right')

    recruit = (tsp['source_in_target'] - tsp['seeded_poison']).clip(lower=0)
    axes[2].plot(tsp['timestep'], tsp['source_in_target'], color=col_p, lw=0.8, alpha=0.5,
                 label='source_in_target (incl. seed)')
    axes[2].plot(tsp['timestep'], recruit, color='black', lw=0.9,
                 label='recruited (seed removed)')
    axes[2].set_ylabel(f'class-{row["source_class"]} entries\nin buffer {target}', fontsize=8)
    axes[2].legend(fontsize=7, loc='upper right')

    j = psc.merge(psp, on='sample_id', suffixes=('_c', '_p'))
    gap = (j['correct_c'].cumsum() - j['correct_p'].cumsum())
    axes[3].plot(j['timestep_p'], gap, color=col_p, lw=1.2)
    axes[3].axhline(0, color='0.4', lw=0.8)
    axes[3].set_ylabel('cumulative accuracy gap\n(clean − poisoned, samples)', fontsize=8)
    axes[3].set_xlabel('timestep')

    for ax in axes:
        if t_ev is not None:
            ax.axvline(t_ev, color='crimson', ls='--', lw=1.2)
    if t_ev is not None:
        axes[0].text(t_ev, axes[0].get_ylim()[1], f'  $t_{{evict}}$={t_ev}', color='crimson',
                     va='top', fontsize=8)
    else:
        axes[0].text(0.02, 0.92, 'seed never evicted (censored)', color='crimson',
                     transform=axes[0].transAxes, fontsize=8)
    gap_end, gap_at_ev = float(gap.iloc[-1]), (float(gap.iloc[t_ev]) if t_ev is not None else np.nan)
    fig.suptitle(FIG_TEXT['fig4_title'].format(
        rule=rule, M=M, target=target, target_name=CLASS_NAMES[target],
        source=int(row['source_class']), source_name=CLASS_NAMES[int(row['source_class'])],
        d_target=row['d_target_acc'], neg=int(row['n_flip_neg']), pos=int(row['n_flip_pos'])),
        y=0.995, fontsize=9)
    savefig(fig, fig_dir, name, log)
    return {'key': row['key'], 'rule': rule, 'M': M, 'target_class': target,
            'source_class': int(row['source_class']), 't_evict': t_ev,
            'censored': bool(row['censored']), 'd_target_acc': float(row['d_target_acc']),
            'd_source_acc': float(row['d_source_acc']),
            'n_flip_neg': int(row['n_flip_neg']), 'n_flip_pos': int(row['n_flip_pos']),
            'recruit_max': int(row['recruit_max']),
            'cum_gap_at_evict': gap_at_ev, 'cum_gap_final': gap_end,
            'gap_widened_after_evict': (bool(gap_end > gap_at_ev) if t_ev is not None else None)}


def fig6_attractor(rl, fig_dir, log):
    """P(pred == target class) in clean vs poisoned runs, per target class."""
    g = rl.groupby('target_class')[['p_pred_target_clean', 'p_pred_target_poisoned']].mean()
    x = np.arange(NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - 0.2, g['p_pred_target_clean'], 0.38, label='clean', color='0.6')
    ax.bar(x + 0.2, g['p_pred_target_poisoned'], 0.38, label='poisoned', color='#0072B2')
    for rule in RULES:
        s = rl[rl['rule'] == rule]
        ax.scatter(s['target_class'] + 0.2, s['p_pred_target_poisoned'], s=9,
                   color=RULE_COLOR[rule], alpha=0.8, zorder=3, edgecolor='none')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}\n{n}' for i, n in enumerate(CLASS_NAMES)], fontsize=6.5)
    ax.set_xlabel('target class c')
    ax.set_ylabel('P(prediction = c)')
    ax.set_title(FIG_TEXT['fig6_attractor_test'], fontsize=8.5)
    ax.legend(fontsize=8)
    savefig(fig, fig_dir, 'fig6_attractor_test', log)


def fig7_flip_timing(flips, fig_dir, log):
    f = flips.dropna(subset=['offset'])
    fig, ax = plt.subplots(figsize=(9, 4))
    if f.empty:
        ax.text(0.5, 0.5, FIG_TEXT['fig7_empty'],
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_axis_off()
    else:
        lim = float(np.percentile(np.abs(f['offset']), 99)) or 1.0
        bins = np.linspace(-lim, lim, 61)
        # step outlines rather than overlaid solid bars: with two filled series the overlap
        # colour is ambiguous and the smaller series is hidden behind the larger one.
        for sign, color, lab in [('negative', '#D55E00', 'negative flips (clean ✓ → poisoned ✗)'),
                                 ('positive', '#009E73', 'positive flips (clean ✗ → poisoned ✓)')]:
            v = f.loc[f['sign'] == sign, 'offset'].clip(-lim, lim)
            ax.hist(v, bins=bins, histtype='step', lw=1.7, color=color, label=lab)
            ax.hist(v, bins=bins, color=color, alpha=0.18)
        ax.axvline(0, color='crimson', ls='--', lw=1.3)
        ax.set_xlabel(FIG_TEXT['fig7_xlabel'])
        ax.set_ylabel('flipped predictions (pooled over evicted runs)')
        ax.legend(fontsize=8)
    ax.set_title(FIG_TEXT['fig7_flip_timing'])
    savefig(fig, fig_dir, 'fig7_flip_timing', log)


# ---------------------------------------------------------------- tables

def table_T1(rl, tab_dir, log):
    cells = _cells(rl)
    data = {}
    for rule, M in cells:
        col = []
        for c in range(NUM_CLASSES):
            s = rl[(rl['rule'] == rule) & (rl['M'] == M) & (rl['target_class'] == c)]
            col.append('—' if s.empty or bool(s.iloc[0]['censored']) else str(int(s.iloc[0]['t_evict'])))
        data[cell_label(rule, M)] = col
    df = pd.DataFrame(data, index=[f'{i} {n}' for i, n in enumerate(CLASS_NAMES)])
    med, cen = {}, {}
    for rule, M in cells:
        s = rl[(rl['rule'] == rule) & (rl['M'] == M)]
        obs = s.loc[~s['censored'], 't_evict']
        med[cell_label(rule, M)] = fmt(float(obs.median()) if len(obs) else np.nan, 1)
        cen[cell_label(rule, M)] = str(int(s['censored'].sum()))
    df.loc['median (evicted only)'] = pd.Series(med)
    df.loc['n censored / 10'] = pd.Series(cen)
    df.index.name = 'target class'
    emit_table(df, 'T1_eviction_times', TABLE_CAPTION['T1_eviction_times'], tab_dir, log)
    return df


def table_T2(rl, steady, tab_dir, log):
    rows = []
    for rule in RULES:
        for M in M_VALUES:
            s = rl[(rl['rule'] == rule) & (rl['M'] == M)]
            if s.empty:
                continue
            obs = s.loc[~s['censored'], 't_evict']
            rows.append({
                'rule': rule, 'M': M, 'n runs': len(s),
                'median t_evict (evicted)': float(obs.median()) if len(obs) else np.nan,
                'n censored': int(s['censored'].sum()),
                'mean Δ overall acc': s['d_overall_acc'].mean(),
                'mean Δ target acc': s['d_target_acc'].mean(),
                'mean Δ source acc': s['d_source_acc'].mean(),
                f'n runs moving target acc > {EFFECT_PT:g}pp': int(
                    (s['d_target_acc'].abs() > EFFECT_PT).sum()),
                'mean excess contam (total, frac)': s['excess_frac_total_contam'].mean(),
                'mean excess contam (target, frac)': s['excess_frac_target_contam'].mean(),
                'mean frac flips after evict': s['frac_flips_after_evict'].mean(),
                'clean contam floor (frac)': steady.get((rule, M), np.nan),
            })
    df = pd.DataFrame(rows).set_index(['rule', 'M'])
    emit_table(df, 'T2_per_rule_summary', TABLE_CAPTION['T2_per_rule_summary'], tab_dir, log)
    return df


def table_T3(rl, inv, tab_dir, log, top_n=3):
    """Flip decomposition for the largest-effect runs."""
    ranked = rl.reindex(rl['d_target_acc'].abs().sort_values(ascending=False).index).head(top_n)
    per_class, sinks = [], []
    for _, r in ranked.iterrows():
        pr = inv[inv['key'] == r['key']].iloc[0]
        cr = inv[(inv['mode'] == 'clean') & (inv['rule'] == r['rule']) & (inv['M'] == r['M'])].iloc[0]
        j = load_ps(cr['persample_path']).merge(load_ps(pr['persample_path']), on='sample_id',
                                                suffixes=('_c', '_p'))
        for c in range(NUM_CLASSES):
            k = j[j['true_class_c'] == c]
            neg = ((k['correct_c'] == 1) & (k['correct_p'] == 0)).sum()
            pos = ((k['correct_c'] == 0) & (k['correct_p'] == 1)).sum()
            per_class.append({'run': r['key'], 'true class': f'{c} {CLASS_NAMES[c]}', 'n': len(k),
                              'correct clean': int(k['correct_c'].sum()),
                              'correct poisoned': int(k['correct_p'].sum()),
                              'neg flips': int(neg), 'pos flips': int(pos),
                              'net': int(k['correct_p'].sum() - k['correct_c'].sum())})
        tc = int(r['target_class'])
        lost = j[(j['true_class_c'] == tc) & (j['correct_c'] == 1) & (j['correct_p'] == 0)]
        if lost.empty:
            sinks.append({'run': r['key'], 'target class': f'{tc} {CLASS_NAMES[tc]}',
                          'lost samples went to': '(no samples lost from the target class)',
                          'count': 0})
        else:
            for pred, n in lost['pred_class_p'].value_counts().items():
                sinks.append({'run': r['key'], 'target class': f'{tc} {CLASS_NAMES[tc]}',
                              'lost samples went to': f'{int(pred)} {CLASS_NAMES[int(pred)]}',
                              'count': int(n)})
    df = pd.DataFrame(per_class).set_index(['run', 'true class'])
    emit_table(df, 'T3_flip_decomposition',
               TABLE_CAPTION['T3_flip_decomposition'].format(top_n=top_n),
               tab_dir, log, float_fmt='%.0f')
    df2 = pd.DataFrame(sinks).set_index(['run', 'target class'])
    emit_table(df2, 'T3b_target_class_sinks', TABLE_CAPTION['T3b_target_class_sinks'], tab_dir, log,
               float_fmt='%.0f')

    # does the lost mass actually go to the poison's source class? (the direct H5 test)
    sink_stats = []
    for _, r in ranked.iterrows():
        s = df2.reset_index()
        s = s[s['run'] == r['key']]
        tot = int(s['count'].sum())
        src = f'{int(r["source_class"])} {CLASS_NAMES[int(r["source_class"])]}'
        to_src = int(s.loc[s['lost samples went to'] == src, 'count'].sum())
        top = s.loc[s['count'].idxmax()] if tot else None
        sink_stats.append({'run': r['key'], 'source_class': src, 'n_lost': tot,
                           'n_lost_to_source_class': to_src,
                           'frac_lost_to_source_class': (to_src / tot) if tot else None,
                           'largest_sink': (top['lost samples went to'] if tot else None),
                           'largest_sink_count': (int(top['count']) if tot else 0)})
    return ranked, df, df2, sink_stats


def table_T4(clean_ts, tab_dir, log):
    rows = []
    for (rule, M), ts in sorted(clean_ts.items()):
        frac = ts['total_contaminant'] / (NUM_CLASSES * M)
        rows.append({'rule': rule, 'M': M, 'in poisoned grid': M in M_VALUES,
                     'mean contaminants': ts['total_contaminant'].mean(),
                     'mean frac': frac.mean(), 'median frac': frac.median(),
                     'steady-state frac (2nd half)': frac.iloc[len(frac) // 2:].mean(),
                     'max frac': frac.max(),
                     'frac of timesteps fully contaminated': float((frac >= 1.0).mean())})
    df = pd.DataFrame(rows).set_index(['rule', 'M'])
    emit_table(df, 'T4_clean_contamination', TABLE_CAPTION['T4_clean_contamination'], tab_dir, log)
    return df


def tables_A(inv, tab_dir, log):
    """A1–A3: accuracy per M, regenerated from the scalar JSONs."""
    out = {}
    for i, M in enumerate(M_VALUES, 1):
        data = {}
        for rule in RULES:
            col = []
            for c in range(NUM_CLASSES):
                s = inv[(inv['mode'] == 'seeded') & (inv['rule'] == rule) & (inv['M'] == M)
                        & (inv['target_class'] == c)]
                col.append(float(s.iloc[0]['overall_acc']) if len(s) else np.nan)
            cl = inv[(inv['mode'] == 'clean') & (inv['rule'] == rule) & (inv['M'] == M)]
            col.append(float(cl.iloc[0]['overall_acc']) if len(cl) else np.nan)
            data[rule] = col
        idx = [f'target {c} ({CLASS_NAMES[c]})' for c in range(NUM_CLASSES)] + ['clean (no poison)']
        df = pd.DataFrame(data, index=idx)
        df.index.name = f'M = {M}'
        name = f'A{i}_accuracy_M{M}'
        emit_table(df, name, TABLE_CAPTION['A_accuracy'].format(M=M), tab_dir, log)
        out[M] = df
    return out


# ---------------------------------------------------------------- summary numbers

def summary_numbers(rl, flips, clean_ts, steady, rho, exemplars, inv_stats, pairing_detail, seeds):
    def pf(x):
        return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)

    S = {'provenance': {'n_poisoned_runs': inv_stats['n_poisoned'],
                        'n_clean_runs': inv_stats['n_clean'],
                        'missing_poisoned': inv_stats['missing_poisoned'],
                        'missing_clean': inv_stats['missing_clean'],
                        'extra_clean_outside_grid': inv_stats['extra_clean'],
                        'pairing_check': pairing_detail, 'seeds_present': seeds}}

    # 1. contamination floor
    floor_rule = {r: pf(np.mean([v for (rr, M), v in steady.items() if rr == r and M in M_VALUES]))
                  for r in RULES}
    S['contamination_floor_clean'] = {
        'per_rule_steady_state_fraction': floor_rule,
        'overall_steady_state_fraction': pf(np.mean([v for k, v in steady.items()
                                                     if k[1] in M_VALUES])),
        'note': descriptions.SUMMARY_NOTE['contamination_floor']}

    # 2. how many runs moved target accuracy
    d = rl['d_target_acc']
    S['effect_counts'] = {f'|delta target acc| > {t}pp': int((d.abs() > t).sum())
                          for t in (1, 5, 10)}
    S['effect_counts'].update({'n_runs': int(len(rl)),
                               'n_runs_target_acc_decreased': int((d < 0).sum()),
                               'n_runs_target_acc_increased': int((d > 0).sum()),
                               'n_runs_target_acc_unchanged': int((d == 0).sum())})

    # 3. survival per rule
    surv = {}
    for r in RULES:
        s = rl[rl['rule'] == r]
        obs = s.loc[~s['censored'], 't_evict']
        surv[r] = {'median_t_evict_evicted_only': pf(obs.median() if len(obs) else np.nan),
                   'median_residency_censored_as_max': pf(s['residency'].median()),
                   'n_censored': int(s['censored'].sum()), 'n_runs': int(len(s))}
    best = max(surv[r]['median_residency_censored_as_max'] for r in RULES)
    tied = [r for r in RULES if surv[r]['median_residency_censored_as_max'] == best]
    longest = tied[0] if len(tied) == 1 else ' and '.join(tied) + ' (tied)'
    fifo_med = surv['fifo']['median_residency_censored_as_max']
    S['survival'] = {'per_rule': surv, 'longest_surviving_rule': longest,
                     'longest_surviving_tied_rules': tied,
                     'ratio_longest_to_fifo_median_residency':
                         pf(best / fifo_med if fifo_med else np.nan),
                     'note': descriptions.SUMMARY_NOTE['survival']}

    # 4. damage after eviction, among runs with a real effect
    eff = rl[(rl['d_target_acc'].abs() > EFFECT_PT) & (~rl['censored'])]
    S['damage_after_eviction'] = {
        'n_runs_considered': int(len(eff)),
        'mean_frac_flips_after_evict': pf(eff['frac_flips_after_evict'].mean()),
        'median_frac_flips_after_evict': pf(eff['frac_flips_after_evict'].median()),
        'definition': descriptions.SUMMARY_NOTE['damage_after_eviction']}

    # 5. attractor direction. Sign counts alone are misleading when the two directions differ
    # in magnitude, so the totals are reported as well.
    att = {}
    for r in RULES:
        s = rl[rl['rule'] == r]
        diff = s['p_pred_target_poisoned'] - s['p_pred_target_clean']
        att[r] = {'n_fell': int((diff < 0).sum()), 'n_rose': int((diff > 0).sum()),
                  'n_unchanged': int((diff == 0).sum()), 'mean_change': pf(diff.mean()),
                  'sum_of_drops': pf(diff[diff < 0].sum()), 'sum_of_rises': pf(diff[diff > 0].sum())}
    all_diff = rl['p_pred_target_poisoned'] - rl['p_pred_target_clean']
    att['ALL'] = {'n_fell': int((all_diff < 0).sum()), 'n_rose': int((all_diff > 0).sum()),
                  'n_unchanged': int((all_diff == 0).sum()), 'mean_change': pf(all_diff.mean()),
                  'sum_of_drops': pf(all_diff[all_diff < 0].sum()),
                  'sum_of_rises': pf(all_diff[all_diff > 0].sum()),
                  'largest_change_key': rl.loc[all_diff.abs().idxmax(), 'key'],
                  'largest_change': pf(all_diff.loc[all_diff.abs().idxmax()])}
    S['attractor_direction'] = att

    # 6. H5 recruitment
    rec = rl['recruit_max'] > 0
    src_up = rl['d_source_acc'] > 0
    S['h5_recruitment'] = {
        'n_runs_with_recruitment': int(rec.sum()), 'n_runs': int(len(rl)),
        'frac_runs_with_recruitment': pf(rec.mean()),
        'n_recruitment_and_source_acc_improved': int((rec & src_up).sum()),
        'n_recruitment_and_source_acc_worsened': int((rec & (rl['d_source_acc'] < 0)).sum()),
        'mean_d_source_acc_when_recruited': pf(rl.loc[rec, 'd_source_acc'].mean()),
        'mean_d_source_acc_when_not_recruited': pf(rl.loc[~rec, 'd_source_acc'].mean()),
        'definition': descriptions.SUMMARY_NOTE['recruitment']}

    # 7. FIFO diagnostic
    f = rl[rl['rule'] == 'fifo']
    zero_flip = int((f['n_flip'] == 0).sum())
    long_resident = f[f['residency'] > 100]
    long_zero = int((long_resident['n_flip'] == 0).sum())
    if zero_flip == len(f) and len(long_resident) > 0 and long_zero == len(long_resident):
        verdict = descriptions.FIFO_VERDICT['suspicious'].format(n_long=len(long_resident))
    elif f['residency'].median() < 50 and zero_flip > 0.5 * len(f):
        verdict = descriptions.FIFO_VERDICT['fast_eviction']
    else:
        verdict = descriptions.FIFO_VERDICT['mixed']
    S['fifo_diagnostic'] = {
        'median_t_evict_evicted_only': pf(f.loc[~f['censored'], 't_evict'].median()),
        'median_residency': pf(f['residency'].median()),
        'mean_n_flip': pf(f['n_flip'].mean()), 'max_n_flip': pf(f['n_flip'].max()),
        'n_runs_zero_flips': zero_flip, 'n_runs': int(len(f)),
        'n_runs_resident_over_100_steps': int(len(long_resident)),
        'n_of_those_with_zero_flips': long_zero, 'verdict': verdict}

    # cross-rule context for the same diagnostic
    S['flip_counts_by_rule'] = {r: {'mean_n_flip': pf(rl[rl['rule'] == r]['n_flip'].mean()),
                                    'n_runs_zero_flips': int((rl[rl['rule'] == r]['n_flip'] == 0).sum()),
                                    'n_runs': int((rl['rule'] == r).sum())} for r in RULES}
    S['spearman_fig3'] = {k: pf(v) if isinstance(v, float) else v for k, v in rho.items()}
    S['exemplars'] = exemplars
    # censored runs have no eviction reference, so the fraction is over evicted runs only
    ev_flips = flips.dropna(subset=['offset']) if len(flips) else flips
    S['flip_pool'] = {'n_flip_events_total': int(len(flips)),
                      'n_in_evicted_runs': int(len(ev_flips)),
                      'n_in_censored_runs': int(len(flips) - len(ev_flips)),
                      'frac_after_eviction': pf((ev_flips['offset'] >= 0).mean())
                      if len(ev_flips) else None}
    return S


# ---------------------------------------------------------------- main

def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw-dir', default=os.path.join(here, 'results', 'runs'),
                   help='directory holding one sub-directory per run')
    p.add_argument('--out-dir', default=os.path.join(here, 'results-inferred'),
                   help='where every derived asset is written')
    p.add_argument('--exemplar', default='similarity,3,1',
                   help='exemplar run for Fig 4 as rule,M,target_class')
    p.add_argument('--reuse-cache', action='store_true',
                   help='reuse derived/run_level.csv and derived/flip_events.csv if present')
    p.add_argument('--dry-run', action='store_true',
                   help='inventory and pairing check only; write nothing')
    args = p.parse_args()

    derived = os.path.join(args.out_dir, 'derived')
    tabs = os.path.join(args.out_dir, 'tables')
    figs = os.path.join(args.out_dir, 'figures')
    if not args.dry_run:
        for d in (derived, tabs, figs):
            os.makedirs(d, exist_ok=True)
    log = Log(None if args.dry_run else os.path.join(args.out_dir, 'run_log.txt'))
    log(f'raw-dir : {args.raw_dir}')
    log(f'out-dir : {args.out_dir}')

    inv = discover_runs(args.raw_dir, log)
    if inv.empty:
        return 1
    inv_stats = check_expected_grid(inv, log)
    seeds = sorted(inv['seed'].dropna().unique().tolist())
    log(f'Seeds present: {seeds}')
    paired_ok, pairing_detail = check_pairing(inv, log)

    if args.dry_run:
        log('\n[dry run] would write:')
        log('  derived/run_level.csv, derived/flip_events.csv, derived/summary_numbers.{json,md},'
            ' derived/file_inventory.txt')
        log('  tables/T1..T4, T3b, A1..A3 (.csv/.tex/.md)')
        log('  figures/fig1..fig7 (.png/.pdf), descriptions.md, run_log.txt')
        return 0

    inv.to_csv(os.path.join(derived, 'file_inventory.txt'), sep='\t', index=False)
    log(f'wrote derived/file_inventory.txt ({len(inv)} rows)')

    rl_path, fl_path = os.path.join(derived, 'run_level.csv'), os.path.join(derived, 'flip_events.csv')
    if args.reuse_cache and os.path.exists(rl_path) and os.path.exists(fl_path):
        rl, flips = pd.read_csv(rl_path), pd.read_csv(fl_path)
        log(f'Reusing cache: {len(rl)} run rows, {len(flips)} flip events.')
    else:
        log('Building run-level table (parsing every ts__/persample__ file)...')
        rl, flips = build_run_level(inv, paired_ok, log)
        rl.to_csv(rl_path, index=False)
        flips.to_csv(fl_path, index=False)
        log(f'wrote derived/run_level.csv ({len(rl)} rows) and derived/flip_events.csv '
            f'({len(flips)} rows)')

    clean_ts = clean_run_frames(inv)

    log('\nFigures:')
    steady = fig5_clean_floor(clean_ts, figs, log)     # first: reframes the whole study
    fig1_survival(rl, figs, log)
    rho, exemplars = {'rho_t_evict': np.nan, 'n_t_evict': 0,
                      'rho_excess_contam': np.nan, 'n_excess_contam': 0}, {}
    vmax = None
    if paired_ok:
        vmax = fig2_heatmaps(rl, figs, log)
        rho = fig3_dose_response(rl, figs, log)
        rule, M, tgt = args.exemplar.split(',')
        exemplars['default'] = fig4_mechanism(rl, inv, 'fig4_mechanism_exemplar', rule.strip(),
                                              int(M), int(tgt), figs, log)
        eff = rl[rl['d_target_acc'].abs() > EFFECT_PT]
        if len(eff):
            mod = eff.iloc[(eff['d_target_acc'].abs() - eff['d_target_acc'].abs().median())
                           .abs().argsort()].iloc[0]
            exemplars['moderate'] = fig4_mechanism(rl, inv, 'fig4b_mechanism_moderate', mod['rule'],
                                                   int(mod['M']), int(mod['target_class']), figs, log)
        else:
            log.warn(f'no run exceeds ±{EFFECT_PT} pp; skipping the moderate-effect exemplar')
        nul = rl.iloc[rl['d_target_acc'].abs().argsort()].iloc[0]
        exemplars['null'] = fig4_mechanism(rl, inv, 'fig4c_mechanism_null', nul['rule'],
                                           int(nul['M']), int(nul['target_class']), figs, log)
        fig6_attractor(rl, figs, log)
        fig7_flip_timing(flips, figs, log)
    else:
        log.warn('paired figures (2, 3, 4, 6, 7) skipped: pairing check failed')

    log('\nTables:')
    T1 = table_T1(rl, tabs, log)
    T4 = table_T4(clean_ts, tabs, log)
    T2 = table_T2(rl, steady, tabs, log) if paired_ok else pd.DataFrame()
    sink_stats = []
    if paired_ok:
        _, _, _, sink_stats = table_T3(rl, inv, tabs, log)
    else:
        log.warn('T2/T3 skipped: pairing check failed')
    tables_A(inv, tabs, log)

    log('\nSummary numbers:')
    S = summary_numbers(rl, flips, clean_ts, steady, rho, exemplars, inv_stats, pairing_detail, seeds)
    S['largest_effect_run_sinks'] = sink_stats
    S['figure_params'] = {'fig2_symmetric_vmax_pp': vmax}
    with open(os.path.join(derived, 'summary_numbers.json'), 'w') as f:
        json.dump(S, f, indent=2)
    descriptions.write_summary_md(S, os.path.join(derived, 'summary_numbers.md'),
                                  lambda v: json.dumps(v, indent=2))
    log('  wrote derived/summary_numbers.{json,md}')
    log(f'  FIFO diagnostic: {S["fifo_diagnostic"]["verdict"]}')

    desc_path = os.path.join(args.out_dir, 'descriptions.md')
    if paired_ok:
        descriptions.write(desc_path, S, rl, T2, T4, exemplars, rho, args.raw_dir, seeds,
                           inv_stats, pairing_detail, RULES, EFFECT_PT)
        log('  wrote descriptions.md')
    else:
        log.warn('descriptions.md written in reduced form (paired analysis unavailable)')
        descriptions.write_unpaired(desc_path, pairing_detail)
    log('\nDone.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
