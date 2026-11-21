"""Nonincreasing piecewise linear curves, including isothermal holds."""
import math
import numpy as np
from src.config import T_START, T_END, DELTA_T


def validate_stages(stages):
    stage_array = np.asarray(stages, dtype=float)
    if stage_array.ndim != 2 or stage_array.shape[1] != 2 or len(stage_array) == 0:
        raise ValueError('each stage requires a duration and a cooling rate')
    durations, rates = stage_array.T
    if not np.isfinite(stage_array).all() or np.any(durations <= 0) or np.any(rates < 0):
        raise ValueError('positive durations and nonnegative cooling rates required')
    if not math.isclose(float(np.sum(durations * rates)), DELTA_T, rel_tol=0, abs_tol=1e-8):
        raise ValueError('stages must end at 20°C: total temperature drop must be 810°C')
    return stage_array


def curve_nodes(stages):
    stage_array = validate_stages(stages)
    times = np.r_[0., np.cumsum(stage_array[:, 0])]
    temperatures = np.r_[T_START, T_START - np.cumsum(stage_array[:, 0]*stage_array[:, 1])]
    temperatures[-1] = T_END  # only float arithmetic (<1e-9), never an invented tail
    return times, temperatures


def canonical_stages(stages):
    stage_array = validate_stages(stages)
    merged = []
    for duration, rate in stage_array:
        if merged and math.isclose(rate, merged[-1][1], rel_tol=1e-12, abs_tol=0):
            merged[-1][0] += float(duration)
        else:
            merged.append([float(duration), float(rate)])
    return merged


def temperature_at(stages, query):
    times, temperatures = curve_nodes(stages)
    return np.interp(query, times, temperatures, left=T_START, right=T_END)


def time_at_temperature(stages, query):
    """First arrival at each temperature; at a hold temperature use its START.

    A temperature just below the hold includes its entire duration. Search on
    decreasing nodes avoids interpolating across duplicate temperatures.
    """
    query_temperatures = np.asarray(query, dtype=float)
    if not np.all((T_END <= query_temperatures) & (query_temperatures <= T_START)):
        raise ValueError('temperature query outside campaign range')
    times, temperatures = curve_nodes(stages)
    result = np.empty(query_temperatures.size)
    for index, temperature in enumerate(query_temperatures.flat):
        right = int(np.searchsorted(-temperatures, -temperature, side='left'))
        if right == 0:
            result[index] = 0.
        else:
            left = right - 1
            fraction = (temperatures[left] - temperature) / (temperatures[left] - temperatures[right])
            result[index] = times[left] + fraction * (times[right] - times[left])
    return result.reshape(query_temperatures.shape)
