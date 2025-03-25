import numpy as np
import numba

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