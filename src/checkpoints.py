"""Atomic training progress files and a fingerprint of the saved ML inputs."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def json_value(value):
    """Use the same representation in memory and after JSON loading (tuples → lists)."""
    return json.loads(json.dumps(value))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, frame):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def dataset_fingerprint(rows):
    """Include labels, curves, group identities and the train/test/CV assignments."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: int(value['curve_id'])):
        values = {key: str(row[key]) for key in
                  ('curve_id', 'group_id', 'status', 'split', 'cv_fold', 'curve_type')}
        values['stages'] = [[float(duration), float(rate)]
                            for duration, rate in json.loads(row['stages_json'])]
        if row['status'] == 'ok':
            values['targets'] = [float(row[phase]) for phase in 'FPB']
        digest.update(json.dumps(values, sort_keys=True, separators=(',', ':')).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def validate_trials(rows, candidate_list):
    if len(rows) > len(candidate_list):
        raise ValueError('more saved trials than planned candidates')
    for index, row in enumerate(rows):
        if int(row['trial']) != index:
            raise ValueError('saved trial indices must be consecutive, without duplicates')
        if json.loads(row['params']) != json_value(candidate_list[index]):
            raise ValueError(f'parameters differ for saved candidate {index + 1}')
        if row.get('status') == 'failed':
            if row.get('cv_MAE_mean') is not None:
                raise ValueError('failed trial has a CV score')
            continue
        scores = json.loads(row['fold_scores'])
        if len(scores) != 5 or not np.isfinite(scores).all():
            raise ValueError('saved trial lacks five finite fold scores')
        if not np.isclose(float(row['cv_MAE_mean']), np.mean(scores), atol=1e-12, rtol=1e-12):
            raise ValueError('saved CV mean differs from fold scores')


def read_trials(path, candidate_list, device='cpu'):
    saved = json.loads(Path(path).read_text())
    if saved.get('device', 'cpu') != device:
        raise ValueError(f'training device differs from checkpoint: {path}')
    if saved['candidates'] != json_value(candidate_list):
        raise ValueError(f'candidate list differs from checkpoint: {path}')
    rows = saved['trials']
    validate_trials(rows, candidate_list)
    return rows


@contextmanager
def training_lock(out):
    path = Path(out) / '.training.lock'
    try:
        with path.open('x') as file:
            file.write(str(os.getpid()))
    except FileExistsError:
        raise ValueError(f'training lock already exists: {path}; check that no training is running') from None
    try:
        yield
    finally:
        path.unlink()
