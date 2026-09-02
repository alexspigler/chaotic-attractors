import numpy as np
import pytest

from chaotic_attractors import estimate_largest_lyapunov, generate_chaotic

TINKERBELL_PARAMS = {"a": 0.9, "b": -0.6013, "c": 2.0, "d": 0.5}


def test_tinkerbell_has_positive_finite_time_exponent():
    x, y = generate_chaotic(
        TINKERBELL_PARAMS,
        "Tinkerbell",
        -0.72,
        -0.64,
        iterations=6000,
        burn_in=1000,
    )

    exponent = estimate_largest_lyapunov(
        x, y, TINKERBELL_PARAMS, "Tinkerbell", max_iterations=5000
    )

    assert exponent == pytest.approx(0.20, abs=0.03)


def test_fixed_point_has_negative_infinite_exponent():
    params = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
    x = np.zeros(100)
    y = np.zeros(100)

    exponent = estimate_largest_lyapunov(x, y, params, "Tinkerbell")

    assert exponent == float("-inf")


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [
        (np.array([0.0]), np.array([0.0]), "at least two"),
        (np.zeros(3), np.zeros(2), "equal length"),
        (np.zeros((2, 2)), np.zeros(4), "one-dimensional"),
    ],
)
def test_rejects_invalid_trajectory_shapes(x, y, message):
    with pytest.raises(ValueError, match=message):
        estimate_largest_lyapunov(x, y, TINKERBELL_PARAMS, "Tinkerbell")
