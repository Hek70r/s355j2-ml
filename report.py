"""Compute the tables and plot inputs from completed thesis experiments."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PHASES
from src.evaluate import paired_diff_ci

VARIANT = ['model', 'representation', 'k', 'mode']
TARGETS = [f'y_{phase}' for phase in PHASES]
PREDICTIONS = [f'yhat_{phase}' for phase in PHASES]


def load_campaign(directory, test):
    runs = pd.read_csv(directory / 'results.csv').query("subset == 'all'").copy()
    runs['tag'] = runs.run.str.rsplit('_seed', n=1).str[0]
    trials = pd.read_csv(directory / 'tuning.csv')
    selection = pd.DataFrame(json.loads((directory / 'selection.json').read_text()))
    metrics = [col for col in runs if col.startswith(('MAE_', 'RMSE_', 'R2_'))]
    by_variant = runs.groupby('tag')
    variants = by_variant[metrics + ['corrected_percent', 'parameters', 'trees']].mean()
    variants = variants.join(by_variant.MAE_mean.agg(['std', 'min', 'max', 'count']).rename(
        columns={'std': 'sd_pp', 'min': 'min_pp', 'max': 'max_pp', 'count': 'seeds'}))
    if by_variant.training_device.nunique().ne(1).any():
        raise ValueError('Inconsistent devices between seeds')
    variants = variants.join(by_variant.training_device.first())
    variants = variants.join(trials.groupby('tag').tuning_s.sum()).join(
        selection.set_index('tag')[VARIANT + ['params', 'cv_MAE_mean', 'final_epochs']],
        how='inner', validate='one_to_one')
    variants['params'] = variants.params.map(json.dumps)
    if set(variants.index) != set(runs.tag) or set(variants.index) != set(selection.tag):
        raise ValueError('Incomplete selections or final fits')
    recommendations = json.loads((directory / 'recommendations_from_cv.json').read_text())
    variants['selected_by_cv'] = variants.index.isin([r['best_tag'] for r in recommendations]) | variants.model.eq('mean')
    variants['candidates'] = trials.groupby('tag').size()
    costs = runs.groupby('model').agg(final_fits=('run', 'size'), fit_s=('fit_s', 'mean'),
                                      predict_s=('predict_s', 'mean'))
    costs['candidates'] = trials.groupby('model').size()
    costs['cv_fits'] = trials.fold_scores.map(lambda value: len(json.loads(value))).groupby(trials.model).sum()
    costs['tuning_hours'] = trials.groupby('model').tuning_s.sum() / 3600
    columns = ['tag', 'seed', 'curve_id', 'group_id', *TARGETS, *PREDICTIONS]
    predictions = pd.read_csv(directory / 'predictions.csv', usecols=columns, float_precision='round_trip')
    predicted_runs = set(zip(predictions.tag, predictions.seed))
    if predicted_runs != set(zip(runs.tag, runs.seed)) or runs.run.duplicated().any():
        raise ValueError('Predictions and final fits differ')
    if predictions.duplicated(['tag', 'seed', 'curve_id']).any() or not predictions.groupby(['tag', 'seed']).size().eq(len(test)).all():
        raise ValueError('Incomplete or duplicated predictions')
    aligned = test.loc[predictions.curve_id]
    if not np.array_equal(predictions.group_id, aligned.group_id):
        raise ValueError('Group assignments differ from the dataset')
    np.testing.assert_allclose(predictions[TARGETS], aligned[list(PHASES)], atol=1e-12, rtol=0)
    predictions['error_pp'] = 100 * np.abs(predictions[PREDICTIONS].to_numpy() - predictions[TARGETS].to_numpy()).mean(axis=1)
    averages = predictions.groupby(['tag', 'seed']).error_pp.mean()
    np.testing.assert_allclose(averages, runs.set_index(['tag', 'seed']).loc[averages.index].MAE_mean, atol=1e-10, rtol=0)
    return variants, costs, predictions


def interval(errors_a, errors_b, test):
    if not errors_a.index.equals(errors_b.index):
        raise ValueError('Paired comparisons require identical curve IDs')
    return paired_diff_ci(errors_a, errors_b, test.loc[errors_a.index].group_id)


def main_comparisons(variants, curve_errors, test):
    chosen = variants[variants.selected_by_cv].sort_values('MAE_mean').index.tolist()
    pairs = [('adjacent_selected', a, b) for a, b in zip(chosen, chosen[1:])]
    for name in ('ridge', 'rf', 'lightgbm', 'mlp', 'cnn1d', 'lstm'):
        mode = 'separate' if name == 'lightgbm' else 'joint'
        pairs.append(('representation_control', f'{name}_Tt_32_{mode}', f'{name}_tT_32_{mode}'))
    for name, rep, mode in [('rf', 'Tt', 'joint'), ('lightgbm', 'Tt', 'separate'), ('cnn1d', 'tT', 'joint')]:
        pairs.append(('K128_minus_K64', f'{name}_{rep}_128_{mode}', f'{name}_{rep}_64_{mode}'))
    for name in ('rf', 'mlp', 'cnn1d', 'lstm'):
        pairs.append(('output_control', f'{name}_tT_32_joint', f'{name}_tT_32_separate'))
    return pd.DataFrame([dict(contrast=kind, tag_a=a, tag_b=b,
                              **interval(curve_errors.loc[a], curve_errors.loc[b], test))
                         for kind, a, b in pairs])


def selected_errors(variants, predictions, test, dataset, metadata, out):
    chosen = variants[variants.selected_by_cv]
    selected = predictions[predictions.tag.isin(chosen.index)].copy()
    selected['model'] = selected.tag.map(chosen.model)
    errors = selected.groupby(['model', 'curve_id']).error_pp.mean().reset_index()
    errors = errors.join(test[['family']], on='curve_id')
    errors['curve_type'] = errors.family.str.split('_').str[0].replace({'sensitivity': 'F3'})
    grouped = errors.groupby('model').error_pp
    distribution = grouped.agg(median_pp='median', max_pp='max')
    distribution['p90_pp'] = grouped.quantile(.90)
    distribution['p99_pp'] = grouped.quantile(.99)
    distribution['gt_1_pct'] = 100 * grouped.apply(lambda values: values.gt(1).mean())
    distribution['le_0_05_count'] = grouped.apply(lambda values: values.le(.05).sum())
    distribution['worst_5pct_share_pct'] = grouped.apply(
        lambda values: 100 * values.nlargest(int(.05 * len(values))).sum() / values.sum())
    distribution.to_csv(out / 'error_distribution.csv')
    errors.groupby(['model', 'curve_type']).error_pp.agg(['mean', 'max', 'count']).to_csv(out / 'families.csv')
    blocks = dataset.family.rename('block').copy()
    for curve_id, family in dataset.family.items():
        if family in ('F3_furnace', 'F3_quench'):
            blocks.loc[curve_id] = f"{family}_band{metadata[str(curve_id)]['params']['hold_band']}"
    blocks = blocks[~blocks.isin(['sensitivity', 'F3_pairs'])]
    counts = blocks.value_counts().rename('n_all').to_frame()
    counts['n_test'] = blocks[blocks.index.isin(test.index)].value_counts()
    block_errors = errors.join(blocks.rename('block'), on='curve_id').groupby(['block', 'model']).error_pp.mean()
    counts.join(block_errors.unstack().drop(columns='mean')).to_csv(out / 'generator_blocks.csv')

    threshold = errors[errors.family.isin(['F3_quench', 'F2_air_quench']) & ~errors.model.isin(['mean', 'ridge'])].copy()
    threshold['temperature_C'] = [metadata[str(r.curve_id)]['params']['hold_C' if r.family == 'F3_quench' else 'switch_C']
                                  for r in threshold.itertuples()]
    threshold[['model', 'curve_id', 'family', 'temperature_C', 'error_pp']].to_csv(out / 'temperature_errors.csv', index=False)
    rf = selected[(selected.model == 'rf') & (selected.seed == 0)].sort_values('curve_id', key=lambda ids: ids.astype(str))
    rf[['curve_id', *PREDICTIONS]].to_csv(out / 'rf_predictions.csv', index=False)

    family_sd = 100 * dataset.groupby('family')[list(PHASES)].std()
    excluded = family_sd.index[family_sd.max(axis=1) < .2]
    kept = selected[~selected.curve_id.isin(test.index[test.family.isin(excluded)])]
    seed_scores = kept.groupby(['model', 'seed']).error_pp.mean()
    summary = seed_scores.groupby('model').agg(['mean', 'std', 'min', 'max']).sort_values('mean')
    summary['n_eval'] = kept.curve_id.nunique()
    summary.to_csv(out / 'nonconstant_models.csv')
    by_model = kept.groupby(['model', 'curve_id']).error_pp.mean()
    comparisons = [dict(model_a=a, model_b=b, **interval(by_model.loc[a], by_model.loc[b], test))
                   for a, b in zip(summary.index, summary.index[1:])]
    pd.DataFrame(comparisons).to_csv(out / 'nonconstant_bootstrap.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/dataset.csv'))
    parser.add_argument('--baseline', type=Path, default=Path('output/main'))
    parser.add_argument('--wide', type=Path)
    parser.add_argument('--softmax', type=Path)
    parser.add_argument('--holdout', type=Path)
    parser.add_argument('--out', type=Path, default=Path('output/tables'))
    args = parser.parse_args()
    dataset = pd.read_csv(args.data, float_precision='round_trip').set_index('curve_id')
    if not dataset.index.is_unique:
        raise ValueError('Duplicate curve IDs in dataset')
    test = dataset[dataset.split == 'test']
    metadata = json.loads(args.data.with_name('generator.json').read_text())
    variants, costs, predictions = load_campaign(args.baseline, test)
    errors = predictions.groupby(['tag', 'curve_id']).error_pp.mean()
    args.out.mkdir(parents=True, exist_ok=True)
    variants.to_csv(args.out / 'variants.csv')
    costs.to_csv(args.out / 'costs.csv')
    main_comparisons(variants, errors, test).to_csv(args.out / 'bootstrap.csv', index=False)
    selected_errors(variants, predictions, test, dataset, metadata, args.out)
    extension_rows = []
    for study, directory in [('wide', args.wide), ('softmax', args.softmax)]:
        if directory is None:
            continue
        extended, costs_extended, predicted = load_campaign(directory, test)
        extension_errors = predicted.groupby(['tag', 'curve_id']).error_pp.mean()
        for tag, row in extended.iterrows():
            mode = 'separate' if row.model == 'lightgbm' else 'joint'
            baseline_tag = f'{row.model}_{row.representation}_{row.k}_{mode}'
            extension_rows.append(dict(study=study, tag=tag, baseline_tag=baseline_tag, **row.to_dict(),
                                       cv_fits=int(costs_extended.loc[row.model, 'cv_fits']),
                                       **interval(extension_errors.loc[tag], errors.loc[baseline_tag], test)))
        if study == 'wide':
            comparison = interval(extension_errors.loc['rf_Tt_128_joint'], extension_errors.loc['lightgbm_Tt_128_separate'], test)
            pd.DataFrame([comparison]).to_csv(args.out / 'wide_rf_minus_lightgbm.csv', index=False)
    if extension_rows:
        pd.DataFrame(extension_rows).to_csv(args.out / 'extensions.csv', index=False)
    if args.holdout:
        holdouts = pd.read_csv(args.holdout / 'results.csv', float_precision='round_trip')
        holdouts[['held_out', 'model', 'n_train', 'n_eval', 'MAE_mean_in_distribution',
                  'MAE_mean', 'delta_MAE_mean', 'ci_low', 'ci_high']].to_csv(args.out / 'holdouts.csv', index=False)


if __name__ == '__main__':
    main()
