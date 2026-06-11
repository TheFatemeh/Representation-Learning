"""
CLIP ViT-B/16 zero-shot evaluation for the cross-dataset (CD) benchmark.
Pure CLIP logits — NO reservoir/TCA adaptation. With --token_pruning it also reproduces the
token-pruning baseline rows of Table 1:
  Ours-0.0 -> CLIP   |   EViT-0.1 -> EViT R=0.9   |   ToME-0.1 -> ToME R=0.9
"""

import argparse
import random

import numpy as np
import torch
from tqdm import tqdm

import clip
from utils import build_test_data_loader, clip_classifier, cls_acc
from measure_gflops import measure_visual_gflops

CD_DATASETS = [
    'caltech101', 'dtd', 'eurosat', 'fgvc', 'food101',
    'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101',
]


def get_arguments():
    parser = argparse.ArgumentParser(description='CLIP zero-shot baseline on CD benchmark')
    parser.add_argument('--datasets', type=str, default='/'.join(CD_DATASETS),
                        help='Datasets separated by /. Default: all 10 CD benchmark datasets.')
    parser.add_argument('--data-root', dest='data_root', type=str, default='data/',
                        help='Path to the datasets directory.')
    parser.add_argument('--backbone', type=str, default='ViT-B/16',
                        help='CLIP backbone (only ViT-B/16 supported).')
    parser.add_argument('--single-template', dest='single_template', action='store_true',
                        help='Use one generic prompt "a photo of a {}." for ALL datasets '
                             '(tests whether the paper CLIP row used a single template).')
    parser.add_argument('--token_pruning', type=str, default='Ours-0.0',
                        help='Visual-encoder token reduction (NO TCA reservoir): Ours-0.0 = none '
                             '(CLIP baseline); EViT-0.1 / ToME-0.1 = EViT/ToME R=0.9 baselines.')
    return parser.parse_args()


def run_zeroshot(loader, clip_model, clip_weights):
    """Standard CLIP zero-shot inference — no adaptation."""
    accuracies = []
    with torch.no_grad():
        for images, target in tqdm(loader, desc='  Inference', leave=False):
            images = images.cuda()
            target = target.cuda()

            # encode_image returns (features, idx, cls_token_list_or_features, cls_token_list)
            image_features, _, _, _ = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features @ clip_weights
            accuracies.append(cls_acc(logits, target))

    return sum(accuracies) / len(accuracies)


def main():
    args = get_arguments()

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    # No TCA reservoir/adaptation — just the (optionally token-reduced) CLIP forward.
    clip_model, preprocess = clip.load(args.backbone, args.token_pruning)
    clip_model.eval()

    gflops = measure_visual_gflops(clip_model)
    print(f'token_pruning: {args.token_pruning}  |  GFLOPs (visual encoder, per image): {gflops:.2f}')

    results = {}
    for dataset_name in args.datasets.split('/'):
        print(f'\n[{dataset_name}]')
        test_loader, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess
        )
        if args.single_template:
            template = ['a photo of a {}.']
        print(f'  prompt: {"single generic" if args.single_template else "official ensemble"}'
              f' ({len(template)} template(s))')
        clip_weights = clip_classifier(classnames, template, clip_model)
        acc = run_zeroshot(test_loader, clip_model, clip_weights)
        results[dataset_name] = acc
        print(f'  Accuracy: {acc:.2f}%')

    print('\n' + '=' * 45)
    print(f'{"Dataset":<22} {"Accuracy":>8}')
    print('-' * 45)
    for name, acc in results.items():
        print(f'{name:<22} {acc:>8.2f}%')
    avg = sum(results.values()) / len(results)
    print('-' * 45)
    print(f'{"Average":<22} {avg:>8.2f}%')
    print('=' * 45)


if __name__ == '__main__':
    main()
