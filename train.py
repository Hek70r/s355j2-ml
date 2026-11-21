#!/usr/bin/env python3
"""Train a selected set of models using the saved dataset split."""
import argparse
import csv
import json
from pathlib import Path
from src.config import load_config, load_training_plan, TrainingPlan, EXTENDED_MODELS, RESOLUTIONS
from src.dataset import read_dataset
from src.experiment import run_experiment, planned_variants, training_manifest, check_resume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/dataset.csv'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', choices=('legacy', 'extended_v1', 'wide_v1'))
    parser.add_argument('--models', nargs='+', choices=EXTENDED_MODELS,
                        help='Train only these models, for example: --models lstm')
    parser.add_argument('--representations', nargs='+', choices=('Tt', 'tT'))
    parser.add_argument('--k', nargs='+', type=int, choices=RESOLUTIONS)
    parser.add_argument('--device', choices=('cpu', 'mps', 'recorded'),
                        help='Neural training device; cpu (default), mps, or recorded for the original mixed-device campaign')
    parser.add_argument('--output-modes', nargs='+', choices=('joint', 'separate', 'softmax'),
                        help='Select output modes; softmax adds four-output neural networks')
    parser.add_argument('--resume', action='store_true',
                        help='Resume the same campaign, reusing completed tuning/refits')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print budgets and fit counts without training or creating files')
    args = parser.parse_args()
    cfg = load_config(args.data.parent/'config.json')
    cfg.require_ready()
    saved = load_training_plan(args.out) if args.resume else TrainingPlan()
    if args.resume and args.device == 'recorded':
        parser.error('Resume uses the device settings already saved in the run directory')
    device = 'mps' if args.device == 'recorded' else args.device or saved.device
    training = TrainingPlan(args.profile or saved.profile,
                            tuple(args.models) if args.models is not None else saved.models,
                            tuple(args.representations or saved.representations),
                            tuple(args.k or saved.resolutions),
                            device,
                            tuple(args.output_modes) if args.output_modes is not None else saved.output_modes)
    try:
        training.validate()
        variants = planned_variants(cfg, training=training)
    except ValueError as error:
        parser.error(str(error))
    manifest = training_manifest(cfg, training, variants)
    summary_keys = ('profile', 'device', 'models', 'representations', 'resolutions', 'output_modes', 'budgets',
                    'variant_count', 'cv_fit_calls', 'final_fit_calls')
    summary = {key: manifest[key] for key in summary_keys}
    print(json.dumps(summary, indent=2), flush=True)
    if args.resume and args.dry_run:
        print(json.dumps(check_resume(args.out, cfg, training, variants,
                                      read_dataset(args.data)), indent=2), flush=True)
    variant_devices = {}
    if args.device == 'recorded':
        if training.profile != 'extended_v1' or 'softmax' in training.output_modes:
            parser.error('recorded applies to the main extended_v1 campaign')
        with (Path(__file__).parent / 'results' / 'variants.csv').open() as source:
            devices = {row['tag']: row['training_device'] for row in csv.DictReader(source)}
        for model, representation, k, mode in variants:
            tag = f'{model}_{representation}_{k}_{mode}'
            variant_devices[tag] = devices[tag]
    if args.dry_run:
        return
    run_experiment(read_dataset(args.data), args.out, cfg, training=training,
                   resume=args.resume, variant_devices=variant_devices)


if __name__ == '__main__':
    main()
