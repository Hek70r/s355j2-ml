"""Sampling ranges and cooling histories for the F1–F4 dataset."""
from itertools import chain

import numpy as np
from scipy.stats import qmc

from src.config import T_START, T_END


# Cooling rates in °C/s, temperatures in °C, hold durations in s.
RATE_BOUNDS = {
    'furnace': (.01, .1), 'air': (.1, 1.), 'forced': (1., 10.),
    'spray': (3., 50.), 'quench': (20., 150.),
}
ROUTES = [('quench', 'air'), ('quench', 'forced'), ('furnace', 'air'),
          ('spray', 'air'), ('air', 'quench')]
HOLD_EDGES = [200., 350., 500., 650., 800.]
F4_SETTINGS = dict(rate_at_600=(.1, 1.), exponent=(1., 1.5),
                   ambient_C=15., temperature_step_C=5.)
CONTROL_TEMPERATURES = [275., 425., 575., 725.]


def sample_range(bounds, coordinate, log=False):
    lower, upper = bounds
    if (not np.isfinite([lower, upper, coordinate]).all() or upper < lower or
            not 0 <= coordinate <= 1 or (log and lower <= 0)):
        raise ValueError('invalid sampling bounds')
    if log:
        value = lower * (upper / lower)**coordinate
    else:
        value = lower + (upper - lower) * coordinate
    return float(value)


def cooling(start_temperature, end_temperature, rate):
    if not np.isfinite([start_temperature, end_temperature, rate]).all() or start_temperature <= end_temperature or rate <= 0:
        raise ValueError('cooling needs decreasing temperatures and a positive rate')
    duration = (start_temperature - end_temperature) / rate
    return [duration, rate]


def smooth_stages(rate_at_600, exponent, step=5., ambient=15.):
    """Integrate dt=-dT/r(T) analytically; then approximate by linear segments.

    r(T)=r600*((T-ambient)/(600-ambient))**p. This is an effective
    shape law, not a spatial heat-transfer model. Ambient <20 permits finite
    arrival at exactly 20 without an arbitrary closing segment.
    """
    if (not np.isfinite([ambient, step, rate_at_600, exponent]).all() or
            not ambient < T_END or step <= 0 or rate_at_600 <= 0 or exponent <= 0):
        raise ValueError('invalid smooth cooling parameters')
    temperatures = np.r_[np.arange(T_START, T_END, -step), T_END]
    start_excess = temperatures[:-1] - ambient
    end_excess = temperatures[1:] - ambient
    if abs(exponent - 1) < 1e-12:
        integral = np.log(start_excess / end_excess)
    else:
        power = 1 - exponent
        integral = (start_excess**power - end_excess**power) / power
    durations = (600-ambient)**exponent / rate_at_600 * integral
    rates = (start_excess - end_excess) / durations
    return np.column_stack((durations, rates)).tolist()


def assemble(first_rate, switch_C=None, second_rate=None, hold_s=0.):
    """Same two-stage construction for switches and controlled holds."""
    if switch_C is None:
        if hold_s != 0:
            raise ValueError('a hold needs a temperature')
        return [cooling(T_START, T_END, first_rate)]
    if (not np.isfinite([switch_C, hold_s]).all() or
            not T_END < switch_C < T_START or hold_s < 0 or second_rate is None):
        raise ValueError('invalid switch/hold')
    stages = [cooling(T_START, switch_C, first_rate)]
    if hold_s:
        stages.append([hold_s, 0.])
    stages.append(cooling(switch_C, T_END, second_rate))
    return stages


def sobol_points(power, seed, dimension, block):
    sampler = qmc.Sobol(dimension, scramble=True, seed=seed + block)
    return sampler.random_base2(power)


def hold_blocks():
    for approach_index, approach in enumerate(('furnace', 'quench')):
        for band, temperatures in enumerate(zip(HOLD_EDGES, HOLD_EDGES[1:])):
            yield 20 + 10 * approach_index + band, 'F3_' + approach, approach, temperatures, (1., 3600.)
    yield 60, 'F3_annealing', 'forced', (600., 720.), (60., 3600.)
    yield 61, 'F3_austempering', 'quench', (450., 550.), (10., 1000.)
    yield 62, 'F3_martempering', 'quench', (420., 450.), (1., 60.)


def sampled_curves(power, seed):
    for block, (regime, bounds) in enumerate(RATE_BOUNDS.items()):
        for (coordinate,) in sobol_points(power, seed, 1, block):
            yield 'F1_' + regime, assemble(sample_range(bounds, coordinate, log=True))

    for block, (before, after) in enumerate(ROUTES, start=10):
        for first, second, switch in sobol_points(power, seed, 3, block):
            first_rate = sample_range(RATE_BOUNDS[before], first, log=True)
            second_rate = sample_range(RATE_BOUNDS[after], second, log=True)
            yield f'F2_{before}_{after}', assemble(first_rate, sample_range((200., 800.), switch), second_rate)

    for block, family, approach, temperatures, durations in hold_blocks():
        for first, second, temperature, duration in sobol_points(power, seed, 4, block):
            stages = assemble(sample_range(RATE_BOUNDS[approach], first, log=True),
                              sample_range(temperatures, temperature),
                              sample_range(RATE_BOUNDS['air'], second, log=True),
                              sample_range(durations, duration, log=True))
            yield family, stages

    for rate, exponent in sobol_points(power, seed, 2, 50):
        yield 'F4', smooth_stages(sample_range(F4_SETTINGS['rate_at_600'], rate, log=True),
                                  sample_range(F4_SETTINGS['exponent'], exponent),
                                  F4_SETTINGS['temperature_step_C'], F4_SETTINGS['ambient_C'])


def control_curves():
    for temperature in CONTROL_TEMPERATURES:
        for duration in (0., 10., 100., 1000.):
            yield 'F3_pairs', assemble(100., temperature, 1., duration)
    for temperature in CONTROL_TEMPERATURES:
        yield 'sensitivity', assemble(100., temperature, 1., 100.)
    for first_rate in (20., 50., 100., 150.):
        yield 'sensitivity', assemble(first_rate, 550., 1., 100.)
    for second_rate in (.1, .3, .6, 1.):
        yield 'sensitivity', assemble(100., 550., second_rate, 100.)


def generate(power, seed):
    if not isinstance(power, int) or not 0 <= power <= 9 or not isinstance(seed, int) or seed < 0:
        raise ValueError('power 0..9 and nonnegative integer seed required')
    # Several controls share the same physical history. Keep its first occurrence.
    seen = set()
    for family, stages in chain(sampled_curves(power, seed), control_curves()):
        signature = tuple((round(duration, 9), round(rate, 9)) for duration, rate in stages)
        if signature not in seen:
            seen.add(signature)
            yield family, stages
