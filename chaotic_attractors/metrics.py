"""Numerical diagnostics for discrete dynamical systems."""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from .core import _get_equation_functions


def estimate_largest_lyapunov(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    params: Mapping[str, float],
    equation_id: str,
    max_iterations: int = 10_000,
    epsilon: float = 1e-7,
) -> float:
    """Estimate the largest Lyapunov exponent along a trajectory.

    The system Jacobian is approximated with centered finite differences. A
    tangent vector is advanced and renormalized at each step, and the returned
    exponent is the mean logarithmic growth per iteration.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if len(x) < 2:
        raise ValueError("at least two trajectory points are required")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number")

    try:
        a, b, c, d = (float(params[name]) for name in ("a", "b", "c", "d"))
    except KeyError as exc:
        raise KeyError(f"Missing required parameter: {exc.args[0]}") from exc

    x_func, y_func = _get_equation_functions(equation_id)
    tangent = np.array([1.0, 1.0], dtype=np.float64)
    tangent /= np.linalg.norm(tangent)
    log_growth = 0.0
    steps = min(len(x) - 1, max_iterations)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for index in range(steps):
            x_value = x[index]
            y_value = y[index]
            x_step = epsilon * max(1.0, abs(x_value))
            y_step = epsilon * max(1.0, abs(y_value))

            jacobian = np.array(
                [
                    [
                        (
                            x_func(x_value + x_step, y_value, a, b, c, d)
                            - x_func(x_value - x_step, y_value, a, b, c, d)
                        )
                        / (2.0 * x_step),
                        (
                            x_func(x_value, y_value + y_step, a, b, c, d)
                            - x_func(x_value, y_value - y_step, a, b, c, d)
                        )
                        / (2.0 * y_step),
                    ],
                    [
                        (
                            y_func(x_value + x_step, y_value, a, b, c, d)
                            - y_func(x_value - x_step, y_value, a, b, c, d)
                        )
                        / (2.0 * x_step),
                        (
                            y_func(x_value, y_value + y_step, a, b, c, d)
                            - y_func(x_value, y_value - y_step, a, b, c, d)
                        )
                        / (2.0 * y_step),
                    ],
                ],
                dtype=np.float64,
            )

            tangent = jacobian @ tangent
            growth = float(np.linalg.norm(tangent))
            if growth == 0.0:
                return float("-inf")
            if not np.isfinite(growth):
                return float("nan")

            log_growth += np.log(growth)
            tangent /= growth

    return float(log_growth / steps)
