"""One campaign configuration. Endpoints are project constants, never tuned."""
from dataclasses import dataclass
import json
from pathlib import Path
import math

T_START = 830.0
T_END = 20.0
DELTA_T = T_START - T_END
RESOLUTIONS = (8, 16, 32, 64, 128)
PHASES = ('F', 'P', 'B', 'M')
SEARCH_SPACES = {
    'ridge': {'alpha': (1e-4, 1e-3, 1e-2, .1, 1., 10., 100., 1000., 10000.)},
    'rf': {'max_depth': (10, 20, None), 'min_samples_leaf': (1, 3, 10), 'max_features': (.5, 1.)},
    'lightgbm': {'num_leaves': (15, 31, 63), 'min_child_samples': (10, 30, 60),
                 'learning_rate': (.03, .15), 'n_estimators': (200, 500, 1000),
                 'reg_lambda': (0., .1, 1., 10.)},
    'mlp': {'hidden': ((32, 16), (64, 32), (128, 64, 32)), 'lr': (1e-4, 3e-3),
            'weight_decay': (1e-6, 1e-2), 'batch_size': (64, 128)},
}

# The dataset Config below is frozen. Training extensions have their own manifest.
NEURAL_MODELS = ('mlp', 'cnn1d', 'lstm')
STOCHASTIC_MODELS = ('rf', 'lightgbm', *NEURAL_MODELS)
LEGACY_MODELS = ('mean', 'ridge', 'rf', 'lightgbm', 'mlp')
EXTENDED_MODELS = (*LEGACY_MODELS, 'cnn1d', 'lstm')
EXTENDED_TRIALS = {'mean': 1, 'ridge': 13, 'rf': 24, 'lightgbm': 36,
                   'mlp': 36, 'cnn1d': 36, 'lstm': 32}
EXTENDED_SEARCH_SPACES = {
    'ridge': {'alpha': tuple(10. ** exponent for exponent in range(-8, 5))},
    'rf': {'max_depth': (5, 10, 20, 40, None),
           'min_samples_leaf': (1, 3, 10, 30), 'max_features': (.3, .5, 1.)},
    'lightgbm': {'num_leaves': (15, 31, 63, 127),
                 'min_child_samples': (5, 10, 30, 60),
                 'learning_rate': (.01, .2), 'n_estimators': (100, 200, 500, 1000),
                 'reg_lambda': (0., .01, .1, 1., 10.)},
    'mlp': {'hidden': ((), (16,), (32,), (64,), (128,), (32, 16), (64, 32),
                       (128, 64, 32), (256, 128, 64)),
            'lr': (1e-5, 1e-2), 'weight_decay': (1e-7, 1e-2),
            'batch_size': (64, 128), 'optimizer_settings': 4},
    'cnn1d': {'channels': ((16,), (32,), (64,), (16, 16), (16, 32), (32, 64)),
              'kernel_size': (3, 5), 'head_width': 32,
              'lr': (1e-5, 1e-2), 'weight_decay': (1e-7, 1e-2),
              'batch_size': (64, 128), 'optimizer_settings': 3},
    'lstm': {'hidden_size': (16, 32, 64, 128), 'num_layers': (1, 2),
             'gradient_clip': 1., 'lr': (1e-5, 1e-2),
             'weight_decay': (1e-7, 1e-2), 'batch_size': (64, 128),
             'optimizer_settings': 4},
}

WIDE_SEARCH_SPACES = {
    'rf': {'max_depth': (20, 40, None), 'min_samples_leaf': (1, 2),
           'max_features': (.1, .2, .3, .5)},
    'lightgbm': {'num_leaves': (63, 127, 255), 'min_child_samples': (1, 2, 5),
                 'learning_rate': (.02, .2), 'n_estimators': (1000, 2000, 4000),
                 'reg_lambda': (10., 30., 100.)},
}
WIDE_TRIALS = {'rf': 24, 'lightgbm': 36}
PROFILE_MODELS = {'legacy': LEGACY_MODELS, 'extended_v1': EXTENDED_MODELS,
                  'wide_v1': ('rf', 'lightgbm')}
PROFILE_SEARCH_SPACES = {'legacy': SEARCH_SPACES, 'extended_v1': EXTENDED_SEARCH_SPACES,
                         'wide_v1': WIDE_SEARCH_SPACES}


@dataclass(frozen=True)
class TrainingPlan:
    profile: str = 'legacy'
    models: tuple[str, ...] = ()
    representations: tuple[str, ...] = ('Tt', 'tT')
    resolutions: tuple[int, ...] = RESOLUTIONS
    device: str = 'cpu'
    output_modes: tuple[str, ...] = ()  # Empty keeps the original experiment matrix.

    def model_names(self):
        available = PROFILE_MODELS[self.profile]
        return self.models or available

    def validate(self):
        if self.device not in ('cpu', 'mps'):
            raise ValueError('training device must be cpu or mps')
        if (len(set(self.output_modes)) != len(self.output_modes)
                or not set(self.output_modes) <= {'joint', 'separate', 'softmax'}):
            raise ValueError('duplicate or unsupported output modes')
        if self.profile not in PROFILE_MODELS:
            raise ValueError('unknown training profile')
        available = PROFILE_MODELS[self.profile]
        for name in self.model_names():
            if name not in available:
                raise ValueError(f'{name} is unavailable in profile {self.profile}')
        for values, allowed in ((self.model_names(), available),
                                (self.representations, ('Tt', 'tT')),
                                (self.resolutions, RESOLUTIONS)):
            if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
                raise ValueError('empty, duplicate or unsupported training selection')


def load_training_plan(results_dir):
    path = Path(results_dir) / 'training.json'
    if not path.exists():
        return TrainingPlan()
    settings = json.loads(path.read_text(encoding='utf-8'))
    plan = TrainingPlan(settings['profile'], tuple(settings['models']),
                        tuple(settings['representations']), tuple(settings['resolutions']),
                        settings.get('device', 'cpu'), tuple(settings.get('output_modes', ())))
    plan.validate()
    return plan


@dataclass(frozen=True)
class Config:
    generator_version: str = 'process_v3'
    power: int = 4  # 2**power Sobol points per sampling block.
    # Fixed bounds cover the designed domain, not extrema learned from test curves.
    time_max_s: float = 120_000.
    rate_max_C_s: float = 400.  # Frozen upper bound used by the time grid.
    seed: int = 0
    final_seeds: tuple[int, ...] = (0, 1, 2)
    time_bins: int = 4
    integration_step_s: float = 0.01
    max_steps: int = 100_000_000
    timeout_s: float = 300
    sum_tolerance: float = 0.002
    reporting_quantum: float = 0.001
    checkpoint_every: int = 20
    rf_trees: int = 400
    search_trials: int = 12
    mlp_max_epochs: int = 500
    patience: int = 30
    min_improvement_pp: float = 0.005
    control_k: int = 32
    bootstrap_replicates: int = 2000
    plateau_epsilon_pp: float = 0.2
    technical: bool = False

    def domain_limits(self):
        return self.time_max_s, self.rate_max_C_s

    def require_ready(self):
        self.validate()
        if self.technical:
            raise ValueError('Training requires a research dataset')

    def validate(self):
        if self.generator_version != 'process_v3':
            raise ValueError('unsupported generator version; archived datasets are separate experiments')
        if not isinstance(self.power, int) or not 0 <= self.power <= 9:
            raise ValueError('power must be an integer from 0 to 9')
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if self.time_bins < 1:
            raise ValueError('time_bins must be positive')
        if not self.final_seeds or len(set(self.final_seeds)) != len(self.final_seeds):
            raise ValueError('final seeds must be nonempty and unique')
        if self.control_k not in RESOLUTIONS:
            raise ValueError('control_k must be a planned resolution')
        budgets = (self.integration_step_s, self.max_steps, self.timeout_s,
                   self.checkpoint_every, self.rf_trees, self.search_trials,
                   self.mlp_max_epochs, self.patience, self.reporting_quantum,
                   self.bootstrap_replicates, self.time_max_s, self.rate_max_C_s)
        if any(not math.isfinite(v) or v <= 0 for v in budgets):
            raise ValueError('numerical and training budgets must be finite and positive')
        if self.time_max_s > self.max_steps * self.integration_step_s:
            raise ValueError('requested duration exceeds Trans_e integration budget')


CONFIG = Config()


def load_config(path):
    values = json.loads(Path(path).read_text(encoding='utf-8'))
    values['final_seeds'] = tuple(values['final_seeds'])
    cfg = Config(**values)
    cfg.validate()
    return cfg
