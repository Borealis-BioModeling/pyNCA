import numpy as np
import numba
from scipy.stats import linregress


@numba.njit(cache=True)
def area_under_curve(t: np.array, c: np.array) -> float:
    """Area under the concentration-time curve (AUC).

    Estimates the area under the concentration time curve
    from t_0 to t_last using integration with the trapezoidal
    method (numpy.trapz).

    Args:
        t (np.array): Times.
        c (np.array): Concentrations.

    Returns:
        float: AUC
    """
    dt = t[1] - t[0]
    return np.trapz(c, t, dx=dt)


@numba.njit(cache=True)
def cmax_tmax(t: np.array, c: np.array) -> tuple[float, float]:
    """Returns the maximum concentration and time of maximum concentration.

    Args:
        t (np.array): Times.
        c (np.array): Concentrations.

    Returns:
        tuple[float, float]: C_max, t_max
    """
    cmax_idx = np.argmax(c)
    cmax = c[cmax_idx]
    tmax = t[cmax_idx]
    return (cmax, tmax)


@numba.njit(cache=True)
def terminal_rate(t: np.array, c: np.array, n_points: int) -> tuple[float, float, float, float]:
    """Returns an estimate of the terminal rate constant and half-life.

    Estimates the terminal rate constant and terminal half-life using
    the specified number of points from the terminal portion of the
    log(concentration)-rate curve.

    Args:
        t (np.array): Times.
        c (np.array): Concentrations.
        n_points (int): Number of terminal points.

    Returns:
        tuple[float, float]: rate_constant, t_half, r_squared, adjusted_r_squared
    """
    slope, intercept, r, p, se = linregress(t, np.log(c))
    r_squared = r**2
    adjusted_r_squared = 1 - (1 - r_squared) * (n_points - 1) / (n_points - 2)
    t_half = np.log(2) / slope
    return (slope, t_half, r_squared, adjusted_r_squared)
