#!/usr/bin/env python3
"""Assemble Table 2 (CIFAR-100-C) from results-t2/*.txt and diff against the paper.

Reads the per-run result files written by the scripts-t2/run_*.sh scripts, extracts the
final top-1 accuracy from each, and prints the paper's Table-2 layout:

    severity x {Contrast, Snow, Brightness} x {CLIP (abs), EViT (delta), Ours (delta)}

EViT/Ours deltas are computed against the 18-template CLIP baseline (all three share that
ensemble). A second block reports the CLIP 80-prompt baseline for comparison.

Run from MainRepo/:   python3 scripts-t2/build_table2.py
"""

import os
import re
import sys

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results-t2')
CORRUPTIONS = ['contrast', 'snow', 'brightness']
SEVERITIES = [1, 2, 3, 4, 5]

# method label -> result-file prefix
METHODS = {
    'CLIP18': 'CLIP',
    'CLIP80': 'CLIP80',
    'EViT': 'EViT',
    'TCA': 'TCA_R0.9',
}

# Paper Table 2 (CLIP absolute; EViT/Ours as delta over CLIP). For side-by-side reference.
PAPER = {
    # severity: {corruption: (CLIP_abs, EViT_delta, Ours_delta)}
    1: {'contrast': (31.90, -3.17, +2.16), 'snow': (35.34, -2.49, +1.50), 'brightness': (41.00, -2.10, +2.10)},
    2: {'contrast': (20.67, +0.68, +3.58), 'snow': (29.72, -1.99, +1.14), 'brightness': (41.44, -1.54, +2.85)},
    3: {'contrast': (15.05, -1.53, +7.04), 'snow': (29.14, -1.68, +1.61), 'brightness': (41.83, -1.82, +2.37)},
    4: {'contrast': (8.85,  -0.34, +11.41), 'snow': (27.04, -2.51, +1.48), 'brightness': (41.12, -1.82, +4.74)},
    5: {'contrast': (2.69,  +1.49, +18.59), 'snow': (24.85, -0.80, +3.30), 'brightness': (38.10, -2.13, +4.75)},
}

ACC_PATTERNS = [
    re.compile(r'Final test accuracy:\s*([0-9.]+)'),   # runner.py (TCA)
    re.compile(r'Accuracy:\s*([0-9.]+)%'),             # clip_zeroshot.py (CLIP/EViT)
]


def read_acc(prefix, corruption, severity):
    path = os.path.join(RESULTS, f'{prefix}_cifar100c-{corruption}-{severity}.txt')
    if not os.path.exists(path):
        return None
    text = open(path).read()
    for pat in ACC_PATTERNS:
        hits = pat.findall(text)
        if hits:
            return float(hits[-1])
    return None


def fmt(x, signed=False):
    if x is None:
        return '   --  '
    return (f'{x:+6.2f}' if signed else f'{x:6.2f}')


def main():
    acc = {m: {} for m in METHODS}
    for m, prefix in METHODS.items():
        for c in CORRUPTIONS:
            for s in SEVERITIES:
                acc[m][(c, s)] = read_acc(prefix, c, s)

    print('\n=== Table 2 — CIFAR-100-C (ours) ===')
    print('Deltas are EViT/TCA minus the 18-template CLIP baseline.\n')
    head = f'{"Sev":>3} |'
    for c in CORRUPTIONS:
        head += f' {c[:8]:>8} CLIP {"EViT":>7} {"Ours":>7} |'
    print(head)
    print('-' * len(head))
    for s in SEVERITIES:
        row = f'{s:>3} |'
        for c in CORRUPTIONS:
            clip = acc['CLIP18'][(c, s)]
            evit = acc['EViT'][(c, s)]
            tca = acc['TCA'][(c, s)]
            d_evit = (evit - clip) if (evit is not None and clip is not None) else None
            d_tca = (tca - clip) if (tca is not None and clip is not None) else None
            row += f' {fmt(clip):>13} {fmt(d_evit, True):>7} {fmt(d_tca, True):>7} |'
        print(row)

    print('\n=== Paper Table 2 (reference) ===')
    print(head)
    print('-' * len(head))
    for s in SEVERITIES:
        row = f'{s:>3} |'
        for c in CORRUPTIONS:
            clip, d_evit, d_tca = PAPER[s][c]
            row += f' {fmt(clip):>13} {fmt(d_evit, True):>7} {fmt(d_tca, True):>7} |'
        print(row)

    print('\n=== CLIP 80-prompt baseline (absolute, for comparison) ===')
    row = f'{"Sev":>3} |'
    print(f'{"Sev":>3} | ' + ' | '.join(f'{c[:10]:>10}' for c in CORRUPTIONS))
    print('-' * 40)
    for s in SEVERITIES:
        cells = [fmt(acc['CLIP80'][(c, s)]) for c in CORRUPTIONS]
        print(f'{s:>3} | ' + ' | '.join(f'{x:>10}' for x in cells))

    missing = [f'{m}:{c}-{s}' for m in METHODS for c in CORRUPTIONS for s in SEVERITIES
               if acc[m][(c, s)] is None]
    if missing:
        print(f'\n[!] {len(missing)} result file(s) missing/unparsed:', ', '.join(missing[:12]),
              '...' if len(missing) > 12 else '')


if __name__ == '__main__':
    sys.exit(main())
