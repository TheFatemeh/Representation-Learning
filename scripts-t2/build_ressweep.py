#!/usr/bin/env python3
"""Analyze the CIFAR-100-C resolution sweep: does TCA's gain over CLIP only appear at low res?

Reads results-t2/<METHOD>_r<RES>_cifar100c-<c>-<s>.txt (plus the full-res 224 baseline files
written without the _r tag), and for each resolution reports CLIP absolute accuracy and the
EViT / TCA deltas over CLIP. Two takeaways:

  1. Which effective resolution reproduces the paper's CLIP column (per-cell error vs PAPER).
  2. Whether the TCA delta grows as resolution drops (the "gains are a low-res artifact" test).

Run from MainRepo/:   python3 scripts-t2/build_ressweep.py
"""

import glob
import os
import re

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results-t2')
CORRUPTIONS = ['contrast', 'snow', 'brightness']
SEVERITIES = [1, 2, 3, 4, 5]

# Paper Table 2 CLIP absolute accuracy (for the "which res matches" comparison).
PAPER_CLIP = {
    'contrast':   {1: 31.90, 2: 20.67, 3: 15.05, 4: 8.85, 5: 2.69},
    'snow':       {1: 35.34, 2: 29.72, 3: 29.14, 4: 27.04, 5: 24.85},
    'brightness': {1: 41.00, 2: 41.44, 3: 41.83, 4: 41.12, 5: 38.10},
}

ACC_PATTERNS = [
    re.compile(r'Final test accuracy:\s*([0-9.]+)'),   # runner.py (TCA)
    re.compile(r'Accuracy:\s*([0-9.]+)%'),             # clip_zeroshot.py (CLIP/EViT)
]


def read_acc(path):
    if not os.path.exists(path):
        return None
    text = open(path).read()
    for pat in ACC_PATTERNS:
        hits = pat.findall(text)
        if hits:
            return float(hits[-1])
    return None


def acc(method_prefix, res, corruption, severity):
    tag = '' if res == 224 else f'_r{res}'
    return read_acc(os.path.join(RESULTS, f'{method_prefix}{tag}_cifar100c-{corruption}-{severity}.txt'))


def discover_resolutions():
    found = set()
    for f in glob.glob(os.path.join(RESULTS, 'CLIP_r*_cifar100c-contrast-1.txt')):
        m = re.search(r'CLIP_r(\d+)_', os.path.basename(f))
        if m:
            found.add(int(m.group(1)))
    if os.path.exists(os.path.join(RESULTS, 'CLIP_cifar100c-contrast-1.txt')):
        found.add(224)
    return sorted(found, reverse=True)


def fmt(x, signed=False):
    if x is None:
        return '  --  '
    return (f'{x:+6.2f}' if signed else f'{x:6.2f}')


def main():
    resolutions = discover_resolutions()
    if not resolutions:
        print('No sweep results found in results-t2/ yet (CLIP_r*_cifar100c-*.txt).')
        return

    print(f'\nResolutions found: {resolutions}  (224 = full-res baseline)\n')

    for c in CORRUPTIONS:
        print(f'=== {c.upper()} — CLIP abs / EViT delta / TCA delta, by resolution ===')
        header = f'{"Sev":>3} | {"paper":>6} |'
        for r in resolutions:
            header += f' res{r:>3}: CLIP {"EViT":>6} {"TCA":>6} |'
        print(header)
        print('-' * len(header))
        for s in SEVERITIES:
            row = f'{s:>3} | {PAPER_CLIP[c][s]:>6.2f} |'
            for r in resolutions:
                clip = acc('CLIP', r, c, s)
                evit = acc('EViT', r, c, s)
                tca = acc('TCA_R0.9', r, c, s)
                d_evit = (evit - clip) if (evit is not None and clip is not None) else None
                d_tca = (tca - clip) if (tca is not None and clip is not None) else None
                row += f' {fmt(clip):>10} {fmt(d_evit, True):>6} {fmt(d_tca, True):>6} |'
            print(row)
        print()

    # Which resolution best reproduces the paper's CLIP column?
    print('=== Mean |CLIP_ours - CLIP_paper| per resolution (lower = closer to paper) ===')
    for r in resolutions:
        errs = [abs(acc('CLIP', r, c, s) - PAPER_CLIP[c][s])
                for c in CORRUPTIONS for s in SEVERITIES
                if acc('CLIP', r, c, s) is not None]
        if errs:
            print(f'  res {r:>3}: mean abs error = {sum(errs)/len(errs):5.2f}  (n={len(errs)})')

    # Does the TCA gain grow as resolution drops? (mean TCA delta over CLIP per resolution)
    print('\n=== Mean TCA delta over CLIP per resolution (the low-res-artifact test) ===')
    for r in resolutions:
        deltas = []
        for c in CORRUPTIONS:
            for s in SEVERITIES:
                clip, tca = acc('CLIP', r, c, s), acc('TCA_R0.9', r, c, s)
                if clip is not None and tca is not None:
                    deltas.append(tca - clip)
        if deltas:
            print(f'  res {r:>3}: mean TCA delta = {sum(deltas)/len(deltas):+5.2f}  (n={len(deltas)})')


if __name__ == '__main__':
    main()
