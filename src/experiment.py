"""Simple select → fit → evaluate flow. All comparisons use one saved split."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import cloudpickle
import numpy as np
import pandas as pd
from src.config import (PHASES, RESOLUTIONS, TrainingPlan, NEURAL_MODELS,
                        STOCHASTIC_MODELS, PROFILE_SEARCH_SPACES,
                        load_config)
from src.dataset import validate_rows, labels
from src.features import prepare_features
from src.models import make_model, capacity, postprocess, candidates
from src.evaluate import tune_model, metrics, correction_stats
from src.checkpoints import (dataset_fingerprint, json_value, read_trials,
                             validate_trials, write_csv, write_json, training_lock)


def planned_variants(cfg, training=None):
    training = training or TrainingPlan()
    training.validate()
    variants = [('mean', 'tT', cfg.control_k, 'joint')]
    for name in training.model_names():
        if name == 'mean':
            continue
        mode = 'separate' if name == 'lightgbm' else 'joint'
        for representation in training.representations:
            for k in training.resolutions:
                variants.append((name, representation, k, mode))
    for name in ('rf', 'mlp'):
        variants.append((name, 'tT', cfg.control_k, 'separate'))
    if 'separate' in training.output_modes:
        # Opt-in controls do not change the matrix of an existing campaign.
        for name in ('cnn1d', 'lstm'):
            variants.append((name, 'tT', cfg.control_k, 'separate'))
    if 'softmax' in training.output_modes:
        for name in NEURAL_MODELS:
            for representation in ('Tt', 'tT'):
                for k in RESOLUTIONS:
                    variants.append((name, representation, k, 'softmax'))
    selected = []
    for name, representation, k, mode in variants:
        if (name in training.model_names() and representation in training.representations
                and k in training.resolutions
                and (not training.output_modes or mode in training.output_modes)):
            selected.append((name, representation, k, mode))
    if not selected:
        raise ValueError('no variants match the model/representation/K selection')
    return selected


def model_filename(run):
    # Tt/tT alone collide on the default case-insensitive macOS filesystem.
    return run.replace('_Tt_', '_temperature_at_time_').replace('_tT_', '_time_to_temperature_') + '.pkl'


def _select_models(stages, train_targets, folds, groups, out, cfg, training, resume=False,
                   variant_devices=None):
    """Reuse completed selections, then checkpoint each remaining candidate."""
    variants = planned_variants(cfg, training)
    has_saved_selection = (out / 'selection.json').exists()
    selections, tuning = read_selection_progress(out, cfg, training, variants) if resume or has_saved_selection else ([], [])
    completed = {selection['tag'] for selection in selections}
    for name, representation, k, mode in variants:
        tag = f'{name}_{representation}_{k}_{mode}'
        device = (variant_devices or {}).get(tag, training.device)
        if tag in completed:
            print('CV already complete:', tag, flush=True)
            continue
        print('CV:', tag, flush=True)
        features, names = prepare_features(stages, representation, k, cfg)
        candidate_list = candidates(name, cfg, training.profile)
        checkpoint = out / 'checkpoints' / model_filename(tag).replace('.pkl', '.json')
        saved_trials = [{key: value for key, value in row.items()
                         if key not in ('tag', 'model', 'representation', 'k', 'mode')}
                        for row in tuning if row['tag'] == tag]
        if resume and saved_trials and not checkpoint.exists():
            write_json(checkpoint, dict(device=device, candidates=candidate_list, trials=saved_trials))
        selected, trials = tune_model(
            name, features, train_targets, folds, groups, cfg, mode, candidate_list,
            profile=training.profile, failure_path=out / f'{tag}_failures.json',
            checkpoint_path=checkpoint, resume=resume, device=device,
        )
        selection = {
            'model': name,
            'representation': representation,
            'k': k,
            'mode': mode,
            'tag': tag,
            'feature_names': names,
            **selected,
        }
        selections.append(selection)
        tuning = [row for row in tuning if row['tag'] != tag]
        for trial in trials:
            tuning.append({
                'tag': tag,
                'model': name,
                'representation': representation,
                'k': k,
                'mode': mode,
                **trial,
            })
        # If interrupted between these writes, full trials can recreate the selection.
        write_csv(out / 'tuning.csv', pd.DataFrame(tuning))
        write_json(out / 'selection.json', selections)
    return selections


def _save_recommendations(selections, out):
    best_by_model = {}
    for selection in selections:
        name = selection['model']
        if name == 'mean' or (name != 'lightgbm' and selection['mode'] == 'separate'):
            continue
        previous = best_by_model.get(name)
        if previous is None or selection['cv_MAE_mean'] < previous['cv_MAE_mean']:
            best_by_model[name] = selection
    write_json(out / 'recommendations_from_cv.json',
               [dict(model=name, best_tag=best['tag']) for name, best in best_by_model.items()])


def _fit_models(selections, stages, train_targets, out, cfg, resume=False, device='cpu',
                variant_devices=None):
    """Refit on all train; a completed saved refit can be reused on resume."""
    fitted = []
    for selection in selections:
        features, _ = prepare_features(stages, selection['representation'], selection['k'], cfg)
        name = selection['model']
        selection_device = (variant_devices or {}).get(selection['tag'], device)
        training_device = selection_device if name in NEURAL_MODELS else 'cpu'
        seeds = cfg.final_seeds if name in STOCHASTIC_MODELS else (cfg.seed,)
        for seed in seeds:
            run = f"{selection['tag']}_seed{seed}"
            model_path = out / 'models' / model_filename(run)
            if resume and model_path.exists():
                with model_path.open('rb') as file:
                    saved = cloudpickle.load(file)
                if (json_value(saved['selection']) != json_value(selection)
                        or saved['seed'] != seed or saved['config'] != cfg
                        or saved.get('training_device', 'cpu') != training_device):
                    raise ValueError(f'saved refit differs from the current selection: {run}')
                fit_s = saved.get('fit_s')
                if fit_s is None and (out / 'results.csv').exists():
                    previous = pd.read_csv(out / 'results.csv')
                    times = previous[(previous.run == run) & (previous.subset == 'all')].fit_s
                    fit_s = float(times.iloc[0]) if len(times) else None
                fitted.append((run, selection, seed, fit_s))
                print('Refit already complete:', run, flush=True)
                del saved
                continue
            print(f"Refit: {selection['tag']}, seed={seed}", flush=True)
            model = make_model(name, selection['params'], selection['mode'], seed, cfg, selection_device)
            started = time.perf_counter()
            if name in NEURAL_MODELS:
                model.fit(features, train_targets, epochs=selection['final_epochs'])
            else:
                model.fit(features, train_targets[:, :3])
            fit_s = time.perf_counter() - started
            temporary = model_path.with_suffix('.pkl.tmp')
            with temporary.open('wb') as file:
                cloudpickle.dump({
                    'model': model,
                    'selection': selection,
                    'seed': seed,
                    'config': cfg,
                    'fit_s': fit_s,
                    'training_device': training_device,
                }, file)
            temporary.replace(model_path)
            fitted.append((run, selection, seed, fit_s))
            del model
    return fitted


def _evaluate_models(fitted, test, out, cfg):
    """Evaluate saved models only after all settings and weights have been saved."""
    print('Final test evaluation', flush=True)
    test_targets = labels(test)
    test_stages = [json.loads(row['stages_json']) for row in test]
    results = []
    predictions = []
    for run, selection, seed, fit_s in fitted:
        with (out / 'models' / model_filename(run)).open('rb') as file:
            saved = cloudpickle.load(file)
        model = saved['model']
        features, _ = prepare_features(test_stages, selection['representation'], selection['k'], cfg)
        started = time.perf_counter()
        raw = model.predict(features)
        predicted_fractions = postprocess(raw)
        predict_s = time.perf_counter() - started
        results.append({
            'run': run,
            'model': selection['model'],
            'representation': selection['representation'],
            'k': selection['k'],
            'mode': selection['mode'],
            'seed': seed,
            'subset': 'all',
            'training_device': saved.get('training_device', 'cpu'),
            'cv_MAE_mean': selection['cv_MAE_mean'],
            'fit_s': fit_s,
            'predict_s': predict_s,
            **capacity(model),
            **metrics(test_targets, predicted_fractions),
            **correction_stats(raw),
        })
        for kind in sorted({row['curve_type'] for row in test}):
            mask = np.array([row['curve_type'] == kind for row in test])
            results.append({
                'run': run,
                'model': selection['model'],
                'representation': selection['representation'],
                'k': selection['k'],
                'mode': selection['mode'],
                'seed': seed,
                'subset': kind,
                **metrics(test_targets[mask], predicted_fractions[mask]),
            })
        for curve_index, row in enumerate(test):
            prediction = {
                'run': run,
                'tag': selection['tag'],
                'model': selection['model'],
                'representation': selection['representation'],
                'k': selection['k'],
                'mode': selection['mode'],
                'seed': seed,
            }
            for key in ('curve_id', 'group_id', 'curve_type'):
                prediction[key] = row[key]
            for phase_index, phase in enumerate(PHASES):
                prediction[f'y_{phase}'] = test_targets[curve_index, phase_index]
                prediction[f'yhat_{phase}'] = predicted_fractions[curve_index, phase_index]
            for phase_index, phase in enumerate('FPB'):
                prediction[f'raw_{phase}'] = raw[curve_index, phase_index]
            if raw.shape[1] == 4:
                prediction['raw_M'] = raw[curve_index, 3]
            predictions.append(prediction)
    write_csv(out / 'results.csv', pd.DataFrame(results))
    write_csv(out / 'predictions.csv', pd.DataFrame(predictions))
    return results


def training_manifest(cfg, training, variants):
    candidate_lists = {}
    for name, _, _, _ in variants:
        if name not in candidate_lists:
            candidate_lists[name] = candidates(name, cfg, training.profile)
    spaces = PROFILE_SEARCH_SPACES[training.profile]
    selected_spaces = {name: spaces.get(name, {}) for name in candidate_lists}
    cv_fits = sum(len(candidate_lists[name]) * 5 for name, _, _, _ in variants)
    final_fits = 0
    for name, _, _, _ in variants:
        final_fits += len(cfg.final_seeds) if name in STOCHASTIC_MODELS else 1
    return dict(profile=training.profile, device=training.device, models=list(candidate_lists),
                output_modes=training.output_modes,
                representations=training.representations, resolutions=training.resolutions,
                variants=variants, search_spaces=selected_spaces, candidates=candidate_lists,
                budgets={name: len(values) for name, values in candidate_lists.items()},
                seed=cfg.seed, final_seeds=cfg.final_seeds, rf_trees=cfg.rf_trees,
                max_epochs=cfg.mlp_max_epochs, patience=cfg.patience,
                min_improvement_pp=cfg.min_improvement_pp,
                optimizer='AdamW', neural_loss=(
                    'MSE on raw F/P/B; MSE on softmax F/P/B/M in softmax mode'
                    if 'softmax' in training.output_modes else 'MSE on raw F/P/B'),
                selection_metric='MAE_mean in percentage points after postprocess',
                variant_count=len(variants), cv_fit_calls=cv_fits, final_fit_calls=final_fits)


def read_selection_progress(out, cfg, training, variants):
    path = out / 'selection.json'
    selections = json.loads(path.read_text()) if path.exists() else []
    path = out / 'tuning.csv'
    if path.exists():
        frame = pd.read_csv(path, float_precision='round_trip')
        tuning = frame.astype(object).where(pd.notna(frame), None).to_dict('records')
    else:
        tuning = []
    expected = {f'{name}_{rep}_{k}_{mode}': (name, rep, k, mode)
                for name, rep, k, mode in variants}
    tags = [row['tag'] for row in selections]
    if len(set(tags)) != len(tags) or not set(tags) <= set(expected):
        raise ValueError('duplicate or unexpected saved selections')
    if any(row['tag'] not in expected for row in tuning):
        raise ValueError('saved tuning contains an unexpected variant')
    for tag, (name, _, _, _) in expected.items():
        rows = sorted((row for row in tuning if row['tag'] == tag), key=lambda row: row['trial'])
        candidate_list = candidates(name, cfg, training.profile)
        validate_trials(rows, candidate_list)
        if tag not in tags:
            continue
        selection = selections[tags.index(tag)]
        actual = tuple(selection[key] for key in ('model', 'representation', 'k', 'mode'))
        if actual != expected[tag] or len(rows) != len(candidate_list):
            raise ValueError(f'incomplete or mismatched saved selection: {tag}')
        valid = [row for row in rows if row.get('status') != 'failed']
        if not valid:
            raise ValueError(f'no valid trials for saved selection: {tag}')
        best = min(valid, key=lambda row: row['cv_MAE_mean'])
        if (json_value(selection['params']) != json.loads(best['params'])
                or selection['cv_MAE_mean'] != best['cv_MAE_mean']):
            raise ValueError(f'saved selection differs from its tuning results: {tag}')
    return selections, tuning


def check_resume(out, cfg, training, variants, rows=None):
    """Read-only validation; dry-run can use this without modifying saved results."""
    out = Path(out)
    if load_config(out / 'config.json') != cfg:
        raise ValueError('cannot resume with a different dataset/training Config')
    saved = json.loads((out / 'training.json').read_text())
    expected = json_value(training_manifest(cfg, training, variants))
    defaults = {'device': 'cpu', 'output_modes': []}
    for key, value in expected.items():
        saved_value = saved.get(key, defaults.get(key))
        if saved_value != value:
            raise ValueError(f'cannot resume: saved training setting differs: {key}')
    if rows is not None and 'dataset_fingerprint' in saved:
        if saved['dataset_fingerprint'] != dataset_fingerprint(rows):
            raise ValueError('cannot resume: dataset content, labels or split changed')
    selections, tuning = read_selection_progress(out, cfg, training, variants)
    variant_devices = saved.get('variant_devices', {})
    tags = {f'{name}_{rep}_{k}_{mode}' for name, rep, k, mode in variants}
    if not set(variant_devices) <= tags or any(value not in ('cpu', 'mps') for value in variant_devices.values()):
        raise ValueError('invalid saved variant devices')
    completed = {row['tag'] for row in selections}
    remaining_fits = 0
    next_tag = None
    for name, rep, k, mode in variants:
        tag = f'{name}_{rep}_{k}_{mode}'
        if tag in completed:
            continue
        next_tag = next_tag or tag
        candidate_list = candidates(name, cfg, training.profile)
        path = out / 'checkpoints' / model_filename(tag).replace('.pkl', '.json')
        device = variant_devices.get(tag, training.device)
        done = len(read_trials(path, candidate_list, device)) if path.exists() else sum(row['tag'] == tag for row in tuning)
        remaining_fits += (len(candidate_list) - done) * 5
    return dict(completed_variants=len(completed), next_variant=next_tag,
                remaining_cv_fit_calls=remaining_fits,
                saved_refits=len(list((out / 'models').glob('*.pkl'))),
                dataset_fingerprint_recorded='dataset_fingerprint' in saved)


def run_experiment(rows, out, cfg, training=None, resume=False, variant_devices=None):
    training = training or TrainingPlan()
    training.validate()
    variants = planned_variants(cfg, training)
    cfg.require_ready()
    validate_rows(rows, cfg, require_splits=True)
    train = [row for row in rows if row['status'] == 'ok' and row['split'] == 'train']
    test = [row for row in rows if row['status'] == 'ok' and row['split'] == 'test']
    out = Path(out)
    if resume:
        progress = check_resume(out, cfg, training, variants, rows)
        print('Resume:', json.dumps(progress), flush=True)
        variant_devices = json.loads((out / 'training.json').read_text()).get('variant_devices', {})
    if any(name in NEURAL_MODELS for name, _, _, _ in variants):
        from src.neural_models import check_device
        for name, representation, k, mode in variants:
            if name in NEURAL_MODELS:
                tag = f'{name}_{representation}_{k}_{mode}'
                check_device((variant_devices or {}).get(tag, training.device))
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
    with training_lock(out):
        (out / 'models').mkdir(exist_ok=True)
        if not resume:
            write_json(out / 'config.json', asdict(cfg))
            manifest = training_manifest(cfg, training, variants)
            manifest['dataset_fingerprint'] = dataset_fingerprint(rows)
            manifest['variant_devices'] = variant_devices or {}
            write_json(out / 'training.json', manifest)
        else:
            manifest = json.loads((out / 'training.json').read_text())
            if 'dataset_fingerprint' not in manifest:
                print('Older checkpoint has no dataset fingerprint; binding to the current saved dataset.', flush=True)
                manifest['dataset_fingerprint'] = dataset_fingerprint(rows)
                write_json(out / 'training.json', manifest)
        train_targets = labels(train)
        folds = np.array([int(row['cv_fold']) for row in train])
        groups = np.array([str(row['group_id']) for row in train])
        stages = [json.loads(row['stages_json']) for row in train]
        variant_devices = manifest.get('variant_devices', {})
        selections = _select_models(stages, train_targets, folds, groups, out, cfg, training,
                                    resume, variant_devices)
        _save_recommendations(selections, out)
        fitted = _fit_models(selections, stages, train_targets, out, cfg, resume, training.device,
                             variant_devices)
        return _evaluate_models(fitted, test, out, cfg)
