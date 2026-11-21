"""F/P/B regression with M-rest, or four neural fractions with softmax."""
import numpy as np
from src.config import CONFIG, PROFILE_MODELS, PROFILE_SEARCH_SPACES, EXTENDED_TRIALS, WIDE_TRIALS


def postprocess(raw):
    """Convert raw F/P/B predictions into nonnegative F/P/B/M fractions.

    The same rule is used for joint and separate regressors, in CV and test.
    A sum below one is preserved: its remainder is the predicted martensite.
    Only sums above one are scaled down. This is not Euclidean projection.
    Four-column softmax outputs already satisfy the constraints and pass through.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] not in (3, 4) or not np.isfinite(raw).all():
        raise ValueError('expected finite predictions, shape (n,3) or (n,4)')
    if raw.shape[1] == 4:
        return raw
    fractions = np.maximum(raw, 0)  # Negative fractions are not allowed.
    total = fractions.sum(axis=1, keepdims=True)
    fractions /= np.maximum(total, 1.)  # Divide by one if the sum is already <= 1.
    martensite = np.maximum(0., 1. - fractions.sum(axis=1))
    return np.column_stack((fractions, martensite))


class MeanModel:
    def fit(self, features, targets):
        self.mean_ = np.mean(targets, axis=0)
        return self

    def predict(self, features):
        return np.tile(self.mean_, (len(features), 1))


class TorchMLP:
    """Joint network or three independent networks, selected as one system."""
    def __init__(self, params, mode, seed, cfg, device='cpu'):
        self.params = params
        self.mode = mode
        self.seed = seed
        self.cfg = cfg
        self.device = device

    gradient_clip = None

    def prepare_tensor(self, features):
        return features

    def build_network(self, input_width, output_width):
        from torch import nn
        layers = []
        for hidden_width in self.params.get('hidden', (64, 32)):
            layers.append(nn.Linear(input_width, hidden_width))
            layers.append(nn.ReLU())
            input_width = hidden_width
        layers.append(nn.Linear(input_width, output_width))
        if self.mode == 'softmax':
            layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)

    def fit(self, features, targets, validation=None, epochs=None):
        from src.neural_models import fit_networks
        return fit_networks(self, features, targets, validation, epochs)

    def predict(self, features):
        from src.neural_models import predict_networks
        return predict_networks(self, features)


def make_model(name, params=None, mode='joint', seed=0, cfg=CONFIG, device='cpu'):
    params = dict(params or {})
    if mode not in ('joint', 'separate', 'softmax'):
        raise ValueError('unknown output mode')
    if mode == 'softmax' and name not in ('mlp', 'cnn1d', 'lstm'):
        raise ValueError('softmax output is only for neural networks')
    if name == 'mean':
        return MeanModel()
    if name == 'ridge':
        if mode != 'joint':
            raise ValueError('Ridge separation is mathematically redundant')
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        return make_pipeline(StandardScaler(), Ridge(alpha=params.get('alpha', 1.)))
    if name == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        base = RandomForestRegressor(
            n_estimators=cfg.rf_trees, random_state=seed, n_jobs=1, **params
        )
        if mode == 'joint':
            return base
        return MultiOutputRegressor(base)
    if name == 'lightgbm':
        if mode != 'separate':
            raise ValueError('LightGBM uses three separate regressors')
        from lightgbm import LGBMRegressor
        from sklearn.multioutput import MultiOutputRegressor
        base = LGBMRegressor(
            random_state=seed,
            verbosity=-1,
            n_jobs=1,
            subsample=1.,
            colsample_bytree=1.,
            **params,
        )
        return MultiOutputRegressor(base)
    if name == 'mlp':
        return TorchMLP(params, mode, seed, cfg, device)
    if name in ('cnn1d', 'lstm'):
        from src.neural_models import TorchSequence
        return TorchSequence(name, params, mode, seed, cfg, device)
    raise ValueError('unknown model')


def candidates(name, cfg, profile='legacy'):
    """Same list for every representation, K, and joint/separate comparison."""
    from itertools import product
    rng = np.random.default_rng(cfg.seed)
    if profile not in PROFILE_MODELS:
        raise ValueError('unknown training profile')
    if name not in PROFILE_MODELS[profile]:
        raise ValueError(f'{name} is unavailable in profile {profile}')
    spaces = PROFILE_SEARCH_SPACES[profile]
    if profile == 'legacy':
        trials = cfg.search_trials
    elif profile == 'extended_v1':
        trials = EXTENDED_TRIALS[name]
    else:
        trials = WIDE_TRIALS[name]

    def draw_log_uniform(low, high):
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))

    if name == 'mean':
        return [{}]
    if name == 'ridge':
        grid = []
        for alpha in spaces['ridge']['alpha']:
            grid.append({'alpha': float(alpha)})
        return grid
    if name == 'rf':
        space = spaces['rf']
        grid = []
        combinations = product(
            space['max_depth'], space['min_samples_leaf'], space['max_features']
        )
        for max_depth, min_samples_leaf, max_features in combinations:
            grid.append(dict(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
            ))
        selected = rng.permutation(len(grid))[:trials]
        return [grid[index] for index in selected]
    if name == 'lightgbm':
        space = spaces['lightgbm']
        sampled = []
        for _ in range(trials):
            num_leaves = int(rng.choice(space['num_leaves']))
            min_child_samples = int(rng.choice(space['min_child_samples']))
            learning_rate = draw_log_uniform(*space['learning_rate'])
            n_estimators = int(rng.choice(space['n_estimators']))
            reg_lambda = float(rng.choice(space['reg_lambda']))
            sampled.append(dict(
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                reg_lambda=reg_lambda,
            ))
        return sampled
    if name in ('mlp', 'cnn1d', 'lstm') and profile == 'extended_v1':
        space = spaces[name]
        optimizer_settings = []
        for index in range(space['optimizer_settings']):
            learning_rate = draw_log_uniform(*space['lr'])
            weight_decay = 0. if index == 0 else draw_log_uniform(*space['weight_decay'])
            batch_size = int(rng.choice(space['batch_size']))
            optimizer_settings.append(dict(lr=learning_rate, weight_decay=weight_decay,
                                           batch_size=batch_size))
        structures = []
        if name == 'mlp':
            for hidden in space['hidden']:
                structures.append(dict(hidden=hidden))
        elif name == 'cnn1d':
            for channels in space['channels']:
                for kernel_size in space['kernel_size']:
                    structures.append(dict(channels=channels, kernel_size=kernel_size,
                                           head_width=space['head_width']))
        else:
            for hidden_size in space['hidden_size']:
                for num_layers in space['num_layers']:
                    structures.append(dict(hidden_size=hidden_size, num_layers=num_layers,
                                           gradient_clip=space['gradient_clip']))
        sampled = []
        # Interleave structures: every round compares the same optimizer settings.
        for settings in optimizer_settings:
            for structure in structures:
                sampled.append({**structure, **settings})
        return sampled
    if name == 'mlp':
        space = spaces['mlp']
        architectures = space['hidden']
        # Default twelve: four candidates per architecture, no K-dependent width.
        sampled = []
        for trial in range(trials):
            hidden = architectures[trial % len(architectures)]
            learning_rate = draw_log_uniform(*space['lr'])
            weight_decay = draw_log_uniform(*space['weight_decay'])
            batch_size = int(rng.choice(space['batch_size']))
            sampled.append(dict(
                hidden=hidden,
                lr=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
            ))
        return sampled
    raise ValueError('unknown model')


def capacity(model):
    if hasattr(model, 'n_parameters_'):
        return dict(parameters=model.n_parameters_, trees=0)
    estimators = getattr(model, 'estimators_', None)
    if isinstance(estimators, list) and estimators and hasattr(estimators[0], 'estimators_'):
        trees = sum(len(estimator.estimators_) for estimator in estimators)
        return dict(parameters=0, trees=trees)
    if isinstance(estimators, list) and estimators and hasattr(estimators[0], 'booster_'):
        trees = sum(estimator.booster_.num_trees() for estimator in estimators)
        return dict(parameters=0, trees=trees)
    if estimators is not None:
        return dict(parameters=0, trees=len(estimators))
    if hasattr(model, 'named_steps'):
        parameters = int(model[-1].coef_.size + np.size(model[-1].intercept_))
        return dict(parameters=parameters, trees=0)
    return dict(parameters=3, trees=0)
