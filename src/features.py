"""K scalar samples of the same simulator input, without resimulation."""
import numpy as np
from src.config import T_START, T_END, DELTA_T, RESOLUTIONS
from src.curves import temperature_at, time_at_temperature


def temperature_grid(k):
    if k not in RESOLUTIONS:
        raise ValueError('K must be 8, 16, 32, 64 or 128')
    grid = T_START - np.arange(1, k+1)*DELTA_T/k
    grid[-1] = T_END
    return grid


def time_grid(k, cfg):
    temperature_grid(k)
    max_time, max_rate = cfg.domain_limits()
    scale = (DELTA_T/128)/max_rate
    return scale*np.expm1(np.arange(1, k+1)/(k+1)*np.log1p(max_time/scale))


def prepare_features(stages_list, representation, k, cfg):
    if representation == 'tT':
        grid = temperature_grid(k)
        feature_matrix = [np.log1p(time_at_temperature(stages, grid)) for stages in stages_list]
        names = [f'log1p_t_to_{temperature:.8g}C' for temperature in grid]
    elif representation == 'Tt':
        grid = time_grid(k, cfg)
        feature_matrix = [temperature_at(stages, grid) for stages in stages_list]
        names = [f'T_at_{time:.8g}s' for time in grid]
    else:
        raise ValueError('only Tt and tT are supported')
    return np.asarray(feature_matrix, dtype=float).reshape(len(stages_list), k), names


def cooling_time_8_5(stages):
    times = time_at_temperature(stages, [800., 500.])
    return float(times[1]-times[0])


def reconstruction_nodes(features, representation, k, cfg):
    if representation == 'tT':
        times = np.r_[0., np.expm1(features)]
        temperatures = np.r_[T_START, temperature_grid(k)]
    elif representation == 'Tt':
        times = np.r_[0., time_grid(k, cfg), cfg.time_max_s]
        temperatures = np.r_[T_START, features, T_END]
    else:
        raise ValueError('only Tt and tT are supported')
    return times, temperatures


def reconstruct_curve(features, representation, k, cfg, query):
    """Inverse reconstruction uses seconds, not interpolation in log(time)."""
    times, temperatures = reconstruction_nodes(features, representation, k, cfg)
    return np.interp(query, times, temperatures, left=T_START, right=T_END)
