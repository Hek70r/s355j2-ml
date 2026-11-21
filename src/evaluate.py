"""Shared five-fold selection using training rows only, and paired evaluation."""
import json
import time
import numpy as np
from src.config import PHASES, NEURAL_MODELS
from src.dataset import inner_split
from src.models import postprocess, make_model, candidates
from src.checkpoints import read_trials, write_json


def metrics(targets, predictions):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    if targets.shape != predictions.shape or targets.ndim != 2 or targets.shape[1] != 4 or len(targets) == 0:
        raise ValueError('metrics require paired nonempty (n,4) arrays')
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError('metrics require finite values')
    error = predictions - targets
    out = {}
    for index, phase in enumerate(PHASES):
        phase_error = error[:, index]
        out['MAE_' + phase] = float(100 * np.mean(np.abs(phase_error)))
        out['RMSE_' + phase] = float(100 * np.sqrt(np.mean(phase_error**2)))
        variance = np.var(targets[:, index])
        if variance > 1e-12:
            out['R2_' + phase] = float(1 - np.mean(phase_error**2) / variance)
        else:
            out['R2_' + phase] = None
    out['MAE_mean'] = float(100 * np.mean(np.abs(error)))
    out['RMSE_mean'] = float(np.mean([out['RMSE_' + phase] for phase in PHASES]))
    return out


def correction_stats(raw):
    raw = np.asarray(raw)
    corrected = postprocess(raw)[:, :raw.shape[1]]
    change = np.abs(corrected - raw)
    corrected_percent = float(100 * np.mean(np.any(change > 1e-12, axis=1)))
    mean_correction_pp = float(100 * np.mean(change))
    return dict(
        corrected_percent=corrected_percent,
        mean_correction_pp=mean_correction_pp,
    )


def tune_model(name, features, targets, folds, groups, cfg, mode='joint', candidate_list=None,
               profile='legacy', failure_path=None, checkpoint_path=None, resume=False, device='cpu'):
    """No test parameter: tuning cannot see test features, labels or IDs."""
    folds = np.asarray(folds, dtype=int)
    groups = np.asarray(groups)
    if set(folds) != set(range(5)):
        raise ValueError('all five saved training folds required')
    for group in set(groups):
        if len(set(folds[groups == group])) != 1:
            raise ValueError('group crosses CV folds')
    rows = []
    best_score = np.inf
    best_params = None
    best_epochs = []
    if candidate_list is None:
        candidate_list = candidates(name, cfg, profile)

    if checkpoint_path is not None:
        from pathlib import Path
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            if not resume:
                raise ValueError(f'checkpoint already exists: {checkpoint_path}; use --resume')
            rows = read_trials(checkpoint_path, candidate_list, device)
            print(f'  {name}: reusing {len(rows)}/{len(candidate_list)} saved candidates', flush=True)
            for row in rows:
                if row.get('status') == 'failed':
                    continue
                if row['cv_MAE_mean'] < best_score:
                    best_score = row['cv_MAE_mean']
                    best_params = json.loads(row['params'])
                    best_epochs = json.loads(row['best_epochs'])

    for trial in range(len(rows), len(candidate_list)):
        params = candidate_list[trial]
        started = time.perf_counter()
        scores = []
        epochs = []
        trained_epochs = []
        error = None
        try:
            for fold in range(5):
                fit_indices = np.flatnonzero(folds != fold)
                validation_indices = np.flatnonzero(folds == fold)
                model = make_model(name, params, mode, cfg.seed, cfg, device)
                if name in NEURAL_MODELS:
                    train_inner, stop_inner = inner_split(groups[fit_indices], cfg.seed + fold)
                    train_indices = fit_indices[train_inner]
                    stop_indices = fit_indices[stop_inner]
                    model.fit(
                        features[train_indices],
                        targets[train_indices],
                        validation=(features[stop_indices], targets[stop_indices]),
                    )
                    epochs.append(model.best_epoch_)
                    trained_epochs.append(getattr(model, 'epochs_trained_', model.best_epoch_))
                else:
                    model.fit(features[fit_indices], targets[fit_indices, :3])
                raw = model.predict(features[validation_indices])
                if not np.isfinite(raw).all():
                    raise FloatingPointError('non-finite CV predictions')
                predictions = postprocess(raw)
                fold_metrics = metrics(targets[validation_indices], predictions)
                scores.append(fold_metrics['MAE_mean'])
        except FloatingPointError as failure:
            if profile == 'legacy':
                raise
            error = str(failure)
        if error is not None:
            rows.append(dict(trial=trial, params=json.dumps(params),
                             cv_MAE_mean=None, cv_std=None,
                             fold_scores=json.dumps(scores), best_epochs=json.dumps(epochs),
                             tuning_s=time.perf_counter() - started,
                             status='failed', error=error))
            if failure_path is not None:
                from pathlib import Path
                path = Path(failure_path)
                failures = json.loads(path.read_text()) if path.exists() else []
                failures.append(dict(model=name, **rows[-1]))
                path.write_text(json.dumps(failures, indent=2))
            if checkpoint_path is not None:
                write_json(checkpoint_path, dict(device=device, candidates=candidate_list, trials=rows))
            print(f'  {name}: candidate {trial+1} failed: {error}', flush=True)
            continue
        score = float(np.mean(scores))
        print(
            f'  {name}: candidate {trial+1}, CV MAE={score:.4f} pp, {time.perf_counter()-started:.1f}s',
            flush=True,
        )
        rows.append(dict(
            trial=trial,
            params=json.dumps(params),
            cv_MAE_mean=score,
            cv_std=float(np.std(scores)),
            fold_scores=json.dumps(scores),
            best_epochs=json.dumps(epochs),
            tuning_s=time.perf_counter() - started,
        ))
        if profile != 'legacy':
            rows[-1].update(status='ok', error='', trained_epochs=json.dumps(trained_epochs))
        if checkpoint_path is not None:
            write_json(checkpoint_path, dict(device=device, candidates=candidate_list, trials=rows))
        if score < best_score:
            best_score = score
            best_params = params
            best_epochs = epochs

    if best_params is None:
        raise ValueError(f'all candidates failed for {name}; failure log: {failure_path}')
    final_epochs = None
    if best_epochs:
        final_epochs = max(1, int(np.median(best_epochs)))
    selected = dict(
        params=best_params,
        cv_MAE_mean=best_score,
        final_epochs=final_epochs,
    )
    return selected, rows


def paired_diff_ci(err_a, err_b, groups, n_boot=2000, seed=0):
    a = np.asarray(err_a)
    b = np.asarray(err_b)
    groups = np.asarray(groups)
    if a.shape != b.shape or a.shape != groups.shape or not len(a):
        raise ValueError('paired errors and groups required')
    diff = a - b
    unique = np.unique(groups)
    totals = np.array([diff[groups == group].sum() for group in unique])
    counts = np.array([np.sum(groups == group) for group in unique])
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for sample in range(n_boot):
        draw = rng.integers(0, len(unique), len(unique))
        boots[sample] = totals[draw].sum() / counts[draw].sum()
    return dict(
        mean=float(diff.mean()),
        ci_low=float(np.quantile(boots, .025)),
        ci_high=float(np.quantile(boots, .975)),
    )
