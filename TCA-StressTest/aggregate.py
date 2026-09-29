#!/usr/bin/env python3
"""Stage 4 aggregator: builds the final metric tables from results/.

Reads every run's persample__*.csv (ground truth for all metrics -- per-class numbers are
recomputed here, so clean and poisoned runs are treated identically) plus
results/identify/poisons.csv (the target->source map).

Writes to results/tables/ and prints to stdout:
  table1_poisons          the 10 identified poisons (target, source/GT, confidence, sample id)
  table2_poison_runs      one row per poisoned run: overall/target/source accuracy+confidence
  table3_clean_runs       one row per clean run: overall acc+conf and per-class accuracy
  table4_classnames       EuroSAT class index -> folder name -> prompt name
  table5_M{1,3,5}         main comparison: per target class x metric, poisoned vs clean vs diff
                          for each update rule.  diff = poisoned - clean (negative => poison hurt).

Conventions: accuracy in %, confidence = mean probability assigned to the GROUND-TRUTH class.
Run:  python3 aggregate.py     (from the TCA-StressTest root; CPU only)
"""
import csv
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT, 'results', 'runs')
POISONS_CSV = os.path.join(ROOT, 'results', 'identify', 'poisons.csv')
OUT_DIR = os.path.join(ROOT, 'results', 'tables')

NUM_CLASSES = 10
CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
PROMPT_NAMES = ['Annual Crop Land', 'Forest', 'Herbaceous Vegetation Land', 'Highway or Road',
                'Industrial Buildings', 'Pasture Land', 'Permanent Crop Land',
                'Residential Buildings', 'River', 'Sea or Lake']
RULES = ['fifo', 'similarity', 'diversity', 'uncertainty']   # main-column order for table 5
MS = [1, 3, 5]                                               # table-5 reservoir sizes


# ---------- loading ----------

def load_run_metrics(run_dir, key):
    """Recompute overall + per-class accuracy/confidence from a run's persample CSV."""
    path = os.path.join(run_dir, f'persample__{key}.csv')
    if not os.path.isfile(path):
        return None
    n = 0
    correct_sum = 0
    conf_sum = 0.0
    cls_n = [0] * NUM_CLASSES
    cls_correct = [0] * NUM_CLASSES
    cls_conf = [0.0] * NUM_CLASSES
    with open(path) as f:
        for row in csv.DictReader(f):
            t = int(row['true_class'])
            c = int(row['correct'])
            g = float(row['conf_gt'])
            n += 1
            correct_sum += c
            conf_sum += g
            cls_n[t] += 1
            cls_correct[t] += c
            cls_conf[t] += g
    return {
        'n': n,
        'overall_acc': 100.0 * correct_sum / n,
        'overall_conf': conf_sum / n,
        'class_acc': [100.0 * cls_correct[k] / cls_n[k] if cls_n[k] else float('nan')
                      for k in range(NUM_CLASSES)],
        'class_conf': [cls_conf[k] / cls_n[k] if cls_n[k] else float('nan')
                       for k in range(NUM_CLASSES)],
    }


def load_all_runs():
    """Return {(rule, M, target): metrics} where target is int for seeded, None for clean."""
    runs = {}
    pat = re.compile(r'^rule-(\w+)__M(\d+)__target-(none|\d+)$')
    for d in sorted(glob.glob(os.path.join(RUNS_DIR, 'rule-*'))):
        key = os.path.basename(d)
        m = pat.match(key)
        if not m:
            print(f"  [warn] unrecognised run dir skipped: {key}")
            continue
        rule, M, tgt = m.group(1), int(m.group(2)), m.group(3)
        target = None if tgt == 'none' else int(tgt)
        metrics = load_run_metrics(d, key)
        if metrics is None:
            print(f"  [warn] missing persample CSV, run skipped: {key}")
            continue
        if metrics['n'] != 8100:
            print(f"  [warn] {key}: only {metrics['n']} samples (expected 8100)")
        runs[(rule, M, target)] = metrics
    return runs


def load_poisons():
    """Return {target_class: {'source': int, 'confidence': float, 'sample_id': int}}."""
    poisons = {}
    with open(POISONS_CSV) as f:
        for row in csv.DictReader(f):
            poisons[int(row['target_class'])] = {
                'source': int(row['source_class']),
                'confidence': float(row['confidence']),
                'sample_id': int(row['sample_id']),
            }
    return poisons


# ---------- output helpers ----------

def write_csv(name, header, rows):
    path = os.path.join(OUT_DIR, name)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {os.path.relpath(path, ROOT)}")
    return path


def print_table(title, header, rows, max_rows=None):
    print(f"\n### {title}")
    widths = [len(h) for h in header]
    shown = rows if max_rows is None else rows[:max_rows]
    for r in shown:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(str(v)))
    fmt = '  '.join('{:>%d}' % w for w in widths)
    print(fmt.format(*header))
    print('  '.join('-' * w for w in widths))
    for r in shown:
        print(fmt.format(*[str(v) for v in r]))
    if max_rows is not None and len(rows) > max_rows:
        print(f"... ({len(rows) - max_rows} more rows in the CSV)")


def fa(x):   # format accuracy
    return f"{x:.2f}"


def fc(x):   # format confidence
    return f"{x:.4f}"


# ---------- tables ----------

def table1_poisons(poisons):
    header = ['target_class', 'target_name', 'source_class(GT)', 'source_name',
              'confidence', 'sample_id']
    rows = []
    for c in sorted(poisons):
        p = poisons[c]
        rows.append([c, CLASSES[c], p['source'], CLASSES[p['source']],
                     f"{p['confidence']:.4f}", p['sample_id']])
    write_csv('table1_poisons.csv', header, rows)
    print_table('Table 1 -- identified poisons (most-confident misclassification per class)',
                header, rows)


def table2_poison_runs(runs, poisons):
    header = ['rule', 'M', 'target_class', 'target_name', 'source_class', 'source_name',
              'overall_acc', 'overall_conf_gt',
              'target_acc', 'target_conf', 'source_acc', 'source_conf']
    rows = []
    for rule in RULES:
        for M in MS:
            for c in range(NUM_CLASSES):
                m = runs.get((rule, M, c))
                if m is None:
                    print(f"  [warn] table2: missing poisoned run {rule}/M{M}/target-{c}")
                    continue
                src = poisons[c]['source']
                rows.append([rule, M, c, CLASSES[c], src, CLASSES[src],
                             fa(m['overall_acc']), fc(m['overall_conf']),
                             fa(m['class_acc'][c]), fc(m['class_conf'][c]),
                             fa(m['class_acc'][src]), fc(m['class_conf'][src])])
    write_csv('table2_poison_runs.csv', header, rows)
    print_table('Table 2 -- per poisoned run (all metrics computable; confidence = prob. on GT class)',
                header, rows, max_rows=12)


def table3_clean_runs(runs):
    header = (['rule', 'M', 'overall_acc', 'overall_conf_gt']
              + [f'acc_c{k}' for k in range(NUM_CLASSES)])
    rows = []
    for (rule, M, target), m in sorted(runs.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if target is not None:
            continue
        rows.append([rule, M, fa(m['overall_acc']), fc(m['overall_conf'])]
                    + [fa(a) for a in m['class_acc']])
    write_csv('table3_clean_runs.csv', header, rows)
    print_table('Table 3 -- clean runs: overall + per-class accuracy', header, rows)


def table4_classnames():
    header = ['index', 'folder_name', 'prompt_name']
    rows = [[k, CLASSES[k], PROMPT_NAMES[k]] for k in range(NUM_CLASSES)]
    write_csv('table4_classnames.csv', header, rows)
    print_table('Table 4 -- EuroSAT class index map', header, rows)


METRICS = [   # (row label, poisoned-run value, clean-run value) given target c and source s
    ('overall_acc',
     lambda m, c, s: m['overall_acc'], lambda m, c, s: m['overall_acc'], fa),
    ('overall_conf_gt',
     lambda m, c, s: m['overall_conf'], lambda m, c, s: m['overall_conf'], fc),
    ('target_class_acc',
     lambda m, c, s: m['class_acc'][c], lambda m, c, s: m['class_acc'][c], fa),
    ('source_class_acc',
     lambda m, c, s: m['class_acc'][s], lambda m, c, s: m['class_acc'][s], fa),
]


def table5_main(runs, poisons):
    for M in MS:
        header = ['target', 'source', 'metric']
        for rule in RULES:
            header += [f'{rule}_poison', f'{rule}_clean', f'{rule}_diff']
        rows = []
        for c in range(NUM_CLASSES):
            s = poisons[c]['source']
            for label, f_pois, f_clean, fmt in METRICS:
                row = [f'c{c} {CLASSES[c]}', f"c'{s} {CLASSES[s]}", label]
                for rule in RULES:
                    pois = runs.get((rule, M, c))
                    clean = runs.get((rule, M, None))
                    if pois is None or clean is None:
                        row += ['NA', 'NA', 'NA']
                        continue
                    pv, cv = f_pois(pois, c, s), f_clean(clean, c, s)
                    row += [fmt(pv), fmt(cv), fmt(pv - cv)]
                rows.append(row)
        write_csv(f'table5_M{M}.csv', header, rows)
        print_table(f'Table 5 -- main comparison, M={M}  '
                    f'(diff = poisoned - clean; negative => poisoning hurt)',
                    header, rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading runs ...")
    runs = load_all_runs()
    poisons = load_poisons()

    n_clean = sum(1 for (_, _, t) in runs if t is None)
    n_pois = sum(1 for (_, _, t) in runs if t is not None)
    print(f"  found {n_clean} clean runs, {n_pois} poisoned runs")
    missing = [(r, M, c) for r in RULES for M in MS for c in range(NUM_CLASSES)
               if (r, M, c) not in runs]
    if missing:
        print(f"  [warn] {len(missing)} poisoned runs missing: {missing}")
    for M in MS:
        for r in RULES:
            if (r, M, None) not in runs:
                print(f"  [warn] clean baseline missing: {r}/M{M}")

    table1_poisons(poisons)
    table2_poison_runs(runs, poisons)
    table3_clean_runs(runs)
    table4_classnames()
    table5_main(runs, poisons)
    print(f"\nAll tables written to {os.path.relpath(OUT_DIR, ROOT)}/")


if __name__ == '__main__':
    main()
