#!/usr/bin/env python3
"""Refit fixed tree configurations after excluding a family or high-temperature holds."""
import argparse
import json
from pathlib import Path
import time


import numpy as np
import pandas as pd

from src.checkpoints import dataset_fingerprint, write_csv
from src.config import PHASES, T_START, load_config
from src.dataset import labels, read_dataset, validate_rows
from src.evaluate import metrics, paired_diff_ci
from src.features import prepare_features
from src.models import make_model, postprocess


BASELINE_TAGS = {'rf': 'rf_Tt_128_joint', 'lightgbm': 'lightgbm_Tt_128_separate'}
MIN_HOLD_TEST_CURVES = 30  # Below this predeclared count also evaluate the 650 C cut.


def hold_temperature(stages):
    temperature = T_START
    for duration, rate in stages:
        if rate == 0 and duration > 0:
            return temperature
        temperature -= duration * rate
    return None


def holdout_cases(train, test):
    """Select by input metadata only; never change the saved train/test split."""
    cases = []
    families = sorted({row['family'] for row in train + test
                       if row['family'] not in ('F3_pairs', 'sensitivity')
                       and (row['family'].startswith(('F1_', 'F2_', 'F3_'))
                            or row['family'] == 'F4')})
    for family in families:
        train_mask = np.array([row['family'] != family for row in train])
        test_mask = np.array([row['family'] == family for row in test])
        if not train_mask.any() or not test_mask.any():
            raise ValueError(f'family {family} has an empty training/evaluation set')
        cases.append(dict(experiment='family', held_out=family,
                          train_mask=train_mask, test_mask=test_mask))

    def above_threshold(row, threshold):
        if row['family'] != 'sensitivity' and not row['family'].startswith('F3_'):
            return False
        temperature = hold_temperature(json.loads(row['stages_json']))
        return temperature is not None and temperature > threshold

    for threshold in (700, 650):
        train_mask = np.array([not above_threshold(row, threshold) for row in train])
        test_mask = np.array([above_threshold(row, threshold) for row in test])
        print(f'Hold > {threshold} C: train={train_mask.sum()}, test={test_mask.sum()}', flush=True)
        if not train_mask.any() or not test_mask.any():
            if threshold == 650:
                raise ValueError('no curves available even for the 650 C hold cut')
        else:
            cases.append(dict(experiment=f'hold_above_{threshold}C', held_out=f'{threshold}C',
                              train_mask=train_mask, test_mask=test_mask))
        if threshold == 700 and test_mask.sum() >= MIN_HOLD_TEST_CURVES:
            break
    return cases


def baseline_predictions(path, test, model_names):
    columns = ['run', 'curve_id', 'group_id', *[f'y_{p}' for p in PHASES],
               *[f'yhat_{p}' for p in PHASES]]
    frame = pd.read_csv(path, usecols=columns, dtype={'curve_id': str, 'group_id': str},
                        float_precision='round_trip')
    curve_ids = [str(row['curve_id']) for row in test]
    truth = labels(test)
    predictions = {}
    for name in model_names:
        run = BASELINE_TAGS[name] + '_seed0'
        selected = frame[frame.run == run].set_index('curve_id')
        if selected.index.duplicated().any() or set(selected.index) != set(curve_ids):
            raise ValueError(f'baseline {run} does not match the saved test curves')
        selected = selected.loc[curve_ids]
        if selected.group_id.tolist() != [str(row['group_id']) for row in test]:
            raise ValueError(f'baseline groups differ: {run}')
        if not np.allclose(selected[[f'y_{p}' for p in PHASES]], truth, atol=1e-12, rtol=0):
            raise ValueError(f'baseline labels differ: {run}')
        predicted = selected[[f'yhat_{p}' for p in PHASES]].to_numpy()
        metrics(truth, predicted)  # Validate shape and finiteness before any refit.
        predictions[name] = predicted
    return predictions


def run_holdouts(rows, results_dir, out, cfg, model_names=('rf', 'lightgbm'), dry_run=False):
    results_dir, out = Path(results_dir), Path(out)
    validate_rows(rows, cfg, require_splits=True)
    if load_config(results_dir / 'config.json') != cfg:
        raise ValueError('baseline configuration differs from the dataset')
    fingerprint = dataset_fingerprint(rows)
    training_manifest = json.loads((results_dir / 'training.json').read_text())
    if training_manifest.get('dataset_fingerprint', fingerprint) != fingerprint:
        raise ValueError('baseline dataset fingerprint differs')
    train = [r for r in rows if r['status'] == 'ok' and r['split'] == 'train']
    test = [r for r in rows if r['status'] == 'ok' and r['split'] == 'test']
    cases = holdout_cases(train, test)
    selections = json.loads((results_dir / 'selection.json').read_text())
    selected_models = {}
    for name in model_names:
        matches = [s for s in selections if s['tag'] == BASELINE_TAGS[name]]
        if len(matches) != 1:
            raise ValueError(f'need exactly one saved selection for {name}')
        selection = matches[0]
        mode = 'joint' if name == 'rf' else 'separate'
        if (selection['model'], selection['representation'], selection['k'], selection['mode']) != (name, 'Tt', 128, mode):
            raise ValueError(f'baseline selection metadata differs: {name}')
        selected_models[name] = selection
    baseline = baseline_predictions(results_dir / 'predictions.csv', test, model_names)
    for case in cases:
        print(case['held_out'], 'train:', case['train_mask'].sum(), 'test:', case['test_mask'].sum())
    if dry_run:
        return
    out.mkdir(parents=True, exist_ok=False)
    train_features, _ = prepare_features([json.loads(r['stages_json']) for r in train], 'Tt', 128, cfg)
    test_features, _ = prepare_features([json.loads(r['stages_json']) for r in test], 'Tt', 128, cfg)
    train_targets, test_targets = labels(train), labels(test)
    results, predictions = [], []
    for case in cases:
        train_mask, test_mask = case['train_mask'], case['test_mask']
        evaluated = [r for r, keep in zip(test, test_mask) if keep]
        for name in model_names:
            print('Fit:', case['experiment'], case['held_out'], name, flush=True)
            selection = selected_models[name]
            model = make_model(name, selection['params'], selection['mode'], seed=0, cfg=cfg)
            started = time.perf_counter()
            model.fit(train_features[train_mask], train_targets[train_mask, :3])
            fit_s = time.perf_counter() - started
            predicted = postprocess(model.predict(test_features[test_mask]))
            del model
            actual, reference = test_targets[test_mask], baseline[name][test_mask]
            errors = 100 * np.abs(predicted - actual).mean(axis=1)
            baseline_errors = 100 * np.abs(reference - actual).mean(axis=1)
            interval = paired_diff_ci(errors, baseline_errors, [r['group_id'] for r in evaluated],
                                      n_boot=cfg.bootstrap_replicates, seed=cfg.seed)
            results.append(dict(experiment=case['experiment'], held_out=case['held_out'],
                                model=name, tag=selection['tag'], mode=selection['mode'], seed=0,
                                n_train=int(train_mask.sum()), n_eval=len(evaluated), fit_s=fit_s,
                                MAE_mean_in_distribution=metrics(actual, reference)['MAE_mean'],
                                delta_MAE_mean=interval['mean'], ci_low=interval['ci_low'],
                                ci_high=interval['ci_high'], **metrics(actual, predicted)))
            for index, row in enumerate(evaluated):
                record = dict(experiment=case['experiment'], held_out=case['held_out'], model=name,
                              mode=selection['mode'], seed=0, representation='Tt', k=128,
                              curve_id=row['curve_id'], family=row['family'], group_id=row['group_id'])
                for column, phase in enumerate(PHASES):
                    record[f'y_{phase}'] = actual[index, column]
                    record[f'yhat_{phase}'] = predicted[index, column]
                    record[f'yhat_in_distribution_{phase}'] = reference[index, column]
                predictions.append(record)
            write_csv(out / 'predictions.csv', pd.DataFrame(predictions))
            write_csv(out / 'results.csv', pd.DataFrame(results))
            print(f"  MAE={results[-1]['MAE_mean']:.4f} pp; baseline="
                  f"{results[-1]['MAE_mean_in_distribution']:.4f} pp; fit={fit_s:.1f}s", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/dataset.csv'))
    parser.add_argument('--results', type=Path, required=True, help='Unmodified baseline campaign')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=tuple(BASELINE_TAGS), default=['rf', 'lightgbm'])
    parser.add_argument('--dry-run', action='store_true', help='Validate pairing and print counts without fitting/writing')
    args = parser.parse_args()
    if len(args.models) != len(set(args.models)):
        parser.error('duplicate models')
    cfg = load_config(args.data.parent / 'config.json')
    cfg.require_ready()
    run_holdouts(read_dataset(args.data), args.results, args.out, cfg, tuple(args.models), args.dry_run)


if __name__ == '__main__':
    main()
