import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import operator
import os
import csv
import json
import copy
import clip
from utils import *
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table

def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='configs/', help='settings of tca on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', default=False, help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, default='oxford_flowers', help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='data/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, default='ViT-B/16', choices=['ViT-B/16'], help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--token_pruning', type=str, default='Ours-0.035', help='pruning method-droprate, parsed as split("-"): name then rate. e.g. Ours-0.035 (R=0.9), EViT-0.1. Must include a rate.')
    parser.add_argument('--visualize_mask', dest='visualize_mask', action='store_true', default=False, help='Save token-condensation visualizations (stock feature; kept off for the stress test).')

    parser.add_argument('--reservoir-sim', dest='reservoir_sim', default=True, help='Using cosine similarity of features as a guidence of caching.')
    parser.add_argument('--div', dest='div', default=True, help='Using the one with small cosine similarity for caching.')
    parser.add_argument('--token_sim', dest='token_sim', default=True, help='Using token-level cosine similarity of features as a guidence of caching.')
    parser.add_argument('--flag', dest='flag', default=True, help='fuse sim cls with current sample.')

    # --- Reservoir-contamination stress-test flags ---
    parser.add_argument('--update_rule', type=str, default='diversity',
                        choices=['fifo', 'uncertainty', 'similarity', 'diversity'],
                        help='Reservoir eviction rule used when a class buffer is full.')
    parser.add_argument('--reservoir_mode', type=str, default='clean',
                        choices=['clean', 'empty', 'seeded'],
                        help="clean: normal TCA. empty: never store anchors (poison-ID run). "
                             "seeded: preload one poison into the target buffer at t=0.")
    parser.add_argument('--reservoir_size', type=int, default=None,
                        help='Override M (reservoir size) from the yaml; used to sweep M in {1,3,5}.')
    parser.add_argument('--target_class', type=int, default=-1,
                        help='Seeded mode: index c of the class buffer that receives the poison.')
    parser.add_argument('--poison_path', type=str, default=None,
                        help='Seeded mode: path to the poison_c{c}.pt seed file to inject.')
    parser.add_argument('--out_dir', type=str, default=None,
                        help="Directory for this run's output files. Auto-named under results/ if omitted.")
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed; also fixes the test-stream order.')

    args = parser.parse_args()

    return args


def _select_evict_index(buffer, update_rule, args):
    """Return the index in `buffer` to drop, per the chosen update rule.

    Item layout: [feature, entropy, cls_token_list, true_label, is_poison, insertion_id].
    Note the field named entropy (item[1]) is the softmax entropy of the sample's CLIP
    logits (low = confident); the rest of the codebase calls this variable `loss`.
    """
    if update_rule == 'fifo':
        # drop the oldest entry (smallest insertion id)
        return int(np.argmin([it[5] for it in buffer]))
    if update_rule == 'uncertainty':
        # drop the most uncertain entry (highest entropy, item[1])
        return int(np.argmax([float(it[1]) for it in buffer]))

    # similarity / diversity: stock cosine-similarity scoring (unchanged math)
    div = (update_rule == 'diversity')
    sim_score = cls_feature_similarity(buffer, args)
    loss = [float(it[1]) for it in buffer]
    if not div:
        sim = [(1 - s) for s in sim_score]
        spread = max(sim) - min(sim)
        weight = int((max(loss) - min(loss)) / spread) if spread != 0 else 0
        score = np.add(loss, [weight * s for s in sim]).tolist()
    else:
        score = np.add(loss, sim_score).tolist()
    return int(torch.argmax(torch.tensor(score)).item())


def update_reservoir_sim(reservoir, pred, item, reservoir_size, update_rule='diversity',
                         clip_model=None, args=None, update_flag=False):
    """Append `item` to its predicted-class buffer and evict per `update_rule` if over capacity."""
    with torch.no_grad():
        if pred in reservoir:
            reservoir[pred].append(item)
        else:
            reservoir[pred] = [item]
        if len(reservoir[pred]) > reservoir_size:
            reservoir[pred].pop(_select_evict_index(reservoir[pred], update_rule, args))

    if update_flag and all(len(reservoir[key]) > 1 for key in reservoir):
        clip_model.update_cls_token(reservoir)
    return clip_model


def compute_reservoir_logits(cls_token_list, scale, reservoir, lambd, beta, clip_weights, neg_mask_thresholds=None, backbone=None):
    """Compute logits using positive/negative reservoir."""
    # exp scaling
    if backbone == "ViT-L/14":
        scaling_weights = np.exp(np.linspace(0, 1, 24) / scale)
    else:
        scaling_weights = np.exp(np.linspace(0, 1, 12) / scale)
    scaling_weights = torch.tensor((scaling_weights / sum(scaling_weights)))
    with torch.no_grad():
        reservoir_keys = []
        reservoir_token_keys = []
        reservoir_values = []
        for class_index in sorted(reservoir.keys()):
            for item in reservoir[class_index]:
                reservoir_keys.append(item[0]) # feature
                reservoir_token_keys.append(item[2].unsqueeze(0))

                reservoir_values.append(class_index)

        reservoir_keys = torch.cat(reservoir_keys, dim=0).permute(1, 0)
        reservoir_token_keys = torch.cat(reservoir_token_keys, dim=0)

        reservoir_values = (F.one_hot(torch.Tensor(reservoir_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        affinity_token = (F.cosine_similarity(cls_token_list.unsqueeze(0), reservoir_token_keys, dim=-1) * scaling_weights).sum(dim=1, keepdim=True).cuda().half()
        affinity = affinity_token.permute(1, 0)
        reservoir_logits = ((-1) * (beta - beta * affinity)).exp() @ reservoir_values
        return lambd * reservoir_logits


def _contamination_counts(reservoir, num_classes, target_class, source_class):
    """Snapshot the reservoir at one timestep.

    A 'contaminant' is a stored entry whose true label != the buffer it sits in.
    Returns (total_contaminants, seeded_poison_count, source_in_target, per_buffer_counts):
      - total              : contaminants summed over all buffers
      - seeded             : entries still flagged is_poison
      - source_in_target   : entries in the targeted buffer c whose true label is the
                             poison's source class c' (the H5 signal: c'-samples pulled into c)
      - per_buffer_counts  : contaminant count for each buffer 0..num_classes-1
    """
    per_buf = []
    total = seeded = source_in_target = 0
    for b in range(num_classes):
        cnt = 0
        for it in reservoir[b]:
            if it[3] != b:                 # true label != buffer -> contaminant
                cnt += 1
            if it[4]:                      # is_poison flag
                seeded += 1
            if b == target_class and it[3] == source_class:
                source_in_target += 1
        per_buf.append(cnt)
        total += cnt
    return total, seeded, source_in_target, per_buf


def _run_key(rule, M, mode, target_class):
    tgt = str(target_class) if mode == 'seeded' else 'none'
    return f"rule-{rule}__M{M}__target-{tgt}"


def _class_metrics(persample_rows, cls):
    """Accuracy (%) and mean ground-truth confidence over samples whose TRUE class == cls.
    persample row = (timestep, sample_id, true_class, pred_class, correct, conf_pred, conf_gt)."""
    rows = [r for r in persample_rows if r[2] == cls]
    if not rows:
        return None, None
    acc = 100.0 * sum(r[4] for r in rows) / len(rows)
    conf = float(np.mean([r[6] for r in rows]))
    return acc, conf


def _write_outputs(args, M, mode, rule, source_class, overall_acc,
                   persample_rows, ts_rows, best_poison, num_classes):
    """Write this run's per-run scalar JSON, per-sample CSV, per-timestep CSV, and
    (empty mode only) the identified poisons.csv + serialized seed tensors."""
    key = _run_key(rule, M, mode, args.target_class)
    out_dir = args.out_dir or (os.path.join('results', 'identify') if mode == 'empty'
                               else os.path.join('results', 'runs', key))
    os.makedirs(out_dir, exist_ok=True)

    # Disaggregated accuracy/confidence on the poisoned (target) class c and the poison's
    # true source class c' (only meaningful for seeded runs; None otherwise).
    tgt_acc, tgt_conf = _class_metrics(persample_rows, args.target_class) if mode == 'seeded' else (None, None)
    src_acc, src_conf = _class_metrics(persample_rows, source_class) if mode == 'seeded' else (None, None)

    scalar = {
        'rule': rule, 'M': M, 'mode': mode,
        'target_class': args.target_class if mode == 'seeded' else None,
        'source_class': source_class if mode == 'seeded' else None,
        'overall_acc': overall_acc,
        'avg_conf_gt': float(np.mean([r[6] for r in persample_rows])),
        'target_class_acc': tgt_acc,
        'target_class_conf': tgt_conf,
        'source_class_acc': src_acc,
        'source_class_conf': src_conf,
        'n_samples': len(persample_rows),
        'seed': args.seed, 'token_pruning': args.token_pruning,
    }
    with open(os.path.join(out_dir, f'scalar__{key}.json'), 'w') as f:
        json.dump(scalar, f, indent=2)

    with open(os.path.join(out_dir, f'persample__{key}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestep', 'sample_id', 'true_class', 'pred_class', 'correct', 'conf_pred', 'conf_gt'])
        w.writerows(persample_rows)

    if ts_rows:
        with open(os.path.join(out_dir, f'ts__{key}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestep', 'total_contaminant', 'seeded_poison', 'source_in_target']
                       + [f'c{c}_contam' for c in range(num_classes)])
            w.writerows(ts_rows)

    if mode == 'empty' and best_poison:
        seed_dir = os.path.join(out_dir, 'poison_seeds')
        os.makedirs(seed_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'poisons.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['target_class', 'source_class', 'confidence', 'sample_id'])
            for c in sorted(best_poison):
                conf, sid, true_c, tensors = best_poison[c]
                w.writerow([c, true_c, conf, sid])
                torch.save({'image_features': tensors[0], 'loss': tensors[1],
                            'cls_token_list': tensors[2], 'source_class': true_c},
                           os.path.join(seed_dir, f'poison_c{c}.pt'))

    print(f"Wrote run outputs to {out_dir}")


def run_test_tca(cfg, loader, clip_model, clip_weights, args):
    with torch.no_grad():
        num_classes = clip_weights.shape[1]
        reservoir, accuracies = {i: [] for i in range(num_classes)}, []

        params = {k: cfg[k] for k in ['reservoir_size', 'scale', 'lambd', 'beta']}
        if args.reservoir_size is not None:
            params['reservoir_size'] = args.reservoir_size
        M = params['reservoir_size']
        mode, rule = args.reservoir_mode, args.update_rule

        clip_model.visual.clip_weights = clip_weights

        # seeded mode: preload one poison into the target buffer at t=0 (insertion id 0 = oldest)
        source_class = -1
        if mode == 'seeded':
            seed = torch.load(args.poison_path, map_location='cpu')
            source_class = int(seed['source_class'])
            # Match the device layout of a naturally-stored item so torch.cat over the buffer
            # works: image_features/loss live on CUDA, but cls_token_list is built on CPU by the
            # encoder (clip/model.py: torch.zeros with no device), so it must stay on CPU here.
            reservoir[args.target_class] = [[seed['image_features'].cuda(), seed['loss'].cuda(),
                                             seed['cls_token_list'], source_class, True, 0]]

        insertion_id = 1  # 0 is reserved for the seeded poison
        persample_rows, ts_rows, best_poison = [], [], {}

        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            image_features, clip_logits, loss, _, pred, cls_token_list = get_clip_logits(images, clip_model, clip_weights)
            target = target.cuda()
            true_class = int(target.item())

            if mode == 'empty':
                final_logits = clip_logits.clone()
            else:
                item = [image_features, loss, cls_token_list, true_class, False, insertion_id]
                insertion_id += 1
                clip_model = update_reservoir_sim(reservoir, pred, item, M, update_rule=rule,
                                                  clip_model=clip_model, args=args, update_flag=args.flag)
                final_logits = clip_logits.clone()
                final_logits += compute_reservoir_logits(cls_token_list, params['scale'], reservoir,
                                                         params['lambd'], params['beta'],
                                                         clip_weights, backbone=args.backbone)

            probs = final_logits.softmax(1)
            pred_final = int(final_logits.topk(1, 1, True, True)[1].t()[0])
            conf_pred, conf_gt = float(probs[0, pred_final]), float(probs[0, true_class])
            accuracies.append(cls_acc(final_logits, target))
            persample_rows.append((i, i, true_class, pred_final, int(pred_final == true_class), conf_pred, conf_gt))

            if mode != 'empty':
                total, seeded, src_in_tgt, per_buf = _contamination_counts(
                    reservoir, num_classes, args.target_class, source_class)
                ts_rows.append([i, total, seeded, src_in_tgt] + per_buf)

            # poison identification (empty mode): keep the most-confident mistake per class
            if mode == 'empty' and pred_final != true_class:
                if pred_final not in best_poison or conf_pred > best_poison[pred_final][0]:
                    best_poison[pred_final] = (conf_pred, i, true_class,
                                               [image_features.cpu(), loss.cpu(), cls_token_list.cpu()])

        overall_acc = sum(accuracies) / len(accuracies)
        print("---- Final test accuracy: {:.2f}. ----\n".format(overall_acc))

        _write_outputs(args, M, mode, rule, source_class, overall_acc,
                       persample_rows, ts_rows, best_poison, num_classes)
        return overall_acc


def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone, args.token_pruning)
    clip_model.eval()

    # Set random seed. Also fixes the shuffled test-stream order: the loader uses shuffle=True
    # over the global RNG (exactly as in the official repo), so a fixed seed reproduces the same
    # order on every run -- identical across clean/seeded modes and all rules/M.
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"

    # Run tca on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")

        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")

        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess, vis_mask=args.visualize_mask)
        clip_weights = clip_classifier(classnames, template, clip_model)

        if args.wandb:
            run_name = f"{dataset_name}" + ""
            run = wandb.init(project="", config=cfg, group=group_name, name=run_name)
        acc = run_test_tca(cfg, test_loader, clip_model, clip_weights, args)

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()

if __name__ == "__main__":
    main()
