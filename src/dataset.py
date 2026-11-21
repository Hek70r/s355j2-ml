"""Published cooling curves, phase fractions and frozen group assignments."""
import csv
import json
from pathlib import Path

import numpy as np
from src.config import CONFIG
from src.curves import canonical_stages, curve_nodes
from src.features import cooling_time_8_5

DATASET_FIELDS = [
    'curve_id', 'family', 'stages_json', 'duration_s', 't8_5_s',
    'F', 'P', 'B', 'M', 'RA', 'status', 'error_reason',
    'group_id', 'split', 'cv_fold',
]

def read_dataset(path):
    with Path(path).open(newline='', encoding='utf-8') as source:
        reader = csv.DictReader(source)
        if not set(DATASET_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Missing dataset columns")
        rows = list(reader)
    for row in rows:
        family = row['family']
        row['curve_type'] = 'F3' if family == 'sensitivity' else family.split('_')[0]
    return rows


def validate_rows(rows, cfg=CONFIG, require_splits=False):
    if not rows:
        raise ValueError('empty dataset')
    ids, curves, groups = set(), set(), {}
    for row in rows:
        key = str(row['curve_id'])
        if not key or key in ids:
            raise ValueError('duplicate curve_id')
        ids.add(key)
        stages = canonical_stages(json.loads(row['stages_json']))
        nodes = curve_nodes(stages)
        # Physical duplicates include redundant subdivision, without using hashes.
        signature = tuple((round(float(t), 10), round(float(T), 10)) for t,T in zip(*nodes))
        if signature in curves:
            raise ValueError('duplicate physical curve')
        curves.add(signature)
        if not np.isclose(float(row['duration_s']), nodes[0][-1], atol=1e-9, rtol=1e-12):
            raise ValueError('wrong duration')
        if not np.isclose(float(row['t8_5_s']), cooling_time_8_5(stages), atol=1e-9, rtol=1e-12):
            raise ValueError('wrong t8/5')
        if nodes[0][-1] > cfg.time_max_s or max(rate for _, rate in stages) > cfg.rate_max_C_s:
            raise ValueError('curve exceeds feature domain')
        if row['status'] == 'ok':
            fractions = np.asarray([float(row[phase]) for phase in 'FPBM'])
            if (not np.isfinite(fractions).all() or np.min(fractions) < -1e-12 or
                    not np.isclose(fractions.sum(), 1., atol=1e-12, rtol=0)):
                raise ValueError(f"Invalid phase fractions for curve {key}")
            if not 0 <= float(row['RA']) < cfg.reporting_quantum / 2:
                raise ValueError(f"Retained austenite outside the accepted range for curve {key}")
            if not row['group_id']:
                raise ValueError('missing group_id')
            if require_splits or row['split']:
                split, fold = row['split'], str(row['cv_fold'])
                if split not in ('train', 'test') or (split == 'train' and fold not in tuple(map(str, range(5)))) or (split == 'test' and fold):
                    raise ValueError('invalid split/fold')
                group = str(row['group_id'])
                if group in groups and groups[group] != (split, fold):
                    raise ValueError('group leakage between split/folds')
                groups[group] = (split, fold)
        elif row['status'] not in ('pending', 'invalid_phases', 'execution_error'):
            raise ValueError('unknown status')
        elif any(row[p] not in ('', None) for p in 'FPB') or row['split'] or str(row['cv_fold']):
            raise ValueError('rejected/pending record has labels or a split')
    if require_splits:
        accepted = [r for r in rows if r['status'] == 'ok']
        if {str(r['cv_fold']) for r in accepted if r['split']=='train'} != set(map(str, range(5))) or not any(r['split']=='test' for r in accepted):
            raise ValueError('need nonempty test and all five training folds')
    return rows


def inner_split(groups, seed):
    from sklearn.model_selection import GroupShuffleSplit
    groups = np.asarray(groups)
    if len(set(groups)) < 2:
        raise ValueError('early stopping requires at least two fitting groups')
    return next(GroupShuffleSplit(n_splits=1, test_size=.1, random_state=seed).split(np.zeros(len(groups)), groups=groups))


def labels(rows):
    """Use the qualified F/P/B fractions and the original remainder rule for M."""
    known = np.asarray([[float(r[p]) for p in 'FPB'] for r in rows])
    return np.column_stack((known, 1-known.sum(axis=1)))
