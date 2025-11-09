"""
Unit tests for equation library definitions.
"""

import numpy as np
import pytest

from chaotic_attractors.core import _get_equation_functions
from chaotic_attractors.equations import EQUATION_LIBRARY


class TestEquationLibrary:
    """Tests for the equation library structure and content."""

    def test_library_not_empty(self):
        """Test that equation library contains equations."""
        assert len(EQUATION_LIBRARY) > 0

    def test_all_equations_have_required_keys(self):
        """Test that all equations have x_eq and y_eq keys."""
        for eq_name, eq_dict in EQUATION_LIBRARY.items():
            assert "x_eq" in eq_dict, f"Missing 'x_eq' in {eq_name}"
            assert "y_eq" in eq_dict, f"Missing 'y_eq' in {eq_name}"

    def test_all_equations_are_strings(self):
        """Test that all equation definitions are strings."""
        for eq_name, eq_dict in EQUATION_LIBRARY.items():
            assert isinstance(eq_dict["x_eq"], str), f"x_eq not a string in {eq_name}"
            assert isinstance(eq_dict["y_eq"], str), f"y_eq not a string in {eq_name}"

    def test_known_equations_exist(self):
        """Test that expected equations are in the library."""
        expected_equations = [
            "Clifford",
            "Tinkerbell",
            "Fractal_Dreams",
            "Peter_Jong",
            "Johnny_Svensson",
            "Custom1",
            "Custom2",
            "Custom3",
            "Custom4",
        ]

        for eq_name in expected_equations:
            assert eq_name in EQUATION_LIBRARY, f"Missing expected equation: {eq_name}"


class TestEquationCompilation:
    """Tests that all equations can be compiled and executed."""

    @pytest.mark.parametrize("equation_id", list(EQUATION_LIBRARY.keys()))
    def test_equation_compiles(self, equation_id):
        """Test that each equation can be compiled without errors."""
        x_func, y_func = _get_equation_functions(equation_id)
        assert callable(x_func)
        assert callable(y_func)

    @pytest.mark.parametrize("equation_id", list(EQUATION_LIBRARY.keys()))
    def test_equation_executes(self, equation_id):
        """Test that each equation can be executed with sample parameters."""
        x_func, y_func = _get_equation_functions(equation_id)

        # Test with sample parameters
        x_prev, y_prev = 0.5, 0.5
        a, b, c, d = 1.0, 1.0, 1.0, 1.0

        try:
            x_next = x_func(x_prev, y_prev, a, b, c, d)
            y_next = y_func(x_prev, y_prev, a, b, c, d)

            # Results should be numeric
            assert isinstance(x_next, (int, float, np.number))
            assert isinstance(y_next, (int, float, np.number))
        except Exception as e:
            pytest.fail(f"Equation {equation_id} failed to execute: {e}")


class TestSpecificEquations:
    """Tests for specific equation behaviors."""

    def test_clifford_with_known_values(self):
        """Test Clifford attractor with known parameters."""
        x_func, y_func = _get_equation_functions("Clifford")

        # Test a single iteration
        x_prev, y_prev = 0.0, 0.0
        a, b, c, d = 1.5, 1.7, 1.0, 0.7

        x_next = x_func(x_prev, y_prev, a, b, c, d)
        y_next = y_func(x_prev, y_prev, a, b, c, d)

        # Expected: x = sin(a*0) + c*cos(a*0) = 0 + 1*1 = 1
        # Expected: y = sin(b*0) + d*cos(b*0) = 0 + 0.7*1 = 0.7
        assert np.isclose(x_next, 1.0)
        assert np.isclose(y_next, 0.7)

    def test_tinkerbell_with_known_values(self):
        """Test Tinkerbell attractor with known parameters."""
        x_func, y_func = _get_equation_functions("Tinkerbell")

        # Test a single iteration
        x_prev, y_prev = 1.0, 1.0
        a, b, c, d = 0.9, -0.6, 2.0, 0.5

        x_next = x_func(x_prev, y_prev, a, b, c, d)
        y_next = y_func(x_prev, y_prev, a, b, c, d)

        # Expected: x = 1^2 - 1^2 + 0.9*1 + (-0.6)*1 = 0 + 0.9 - 0.6 = 0.3
        # Expected: y = 2*1*1 + 2.0*1 + 0.5*1 = 2 + 2 + 0.5 = 4.5
        assert np.isclose(x_next, 0.3)
        assert np.isclose(y_next, 4.5)

    def test_fractal_dreams_with_known_values(self):
        """Test Fractal_Dreams attractor with known parameters."""
        x_func, y_func = _get_equation_functions("Fractal_Dreams")

        # Test a single iteration
        x_prev, y_prev = 1.0, 1.0
        a, b, c, d = 0.5, 0.5, 1.0, 1.0

        x_next = x_func(x_prev, y_prev, a, b, c, d)
        y_next = y_func(x_prev, y_prev, a, b, c, d)

        # Expected: x = sin(0.5*1.0) + 1.0*sin(0.5*1.0)
        # Expected: y = sin(0.5*1.0) + 1.0*sin(0.5*1.0)
        assert np.isclose(x_next, 2 * np.sin(0.5))
        assert np.isclose(y_next, 2 * np.sin(0.5))

    def test_peter_jong_with_known_values(self):
        """Test Peter_Jong attractor with known parameters."""
        x_func, y_func = _get_equation_functions("Peter_Jong")

        # Test a single iteration
        x_prev, y_prev = 1.0, 1.0
        a, b, c, d = 1.0, 1.0, 1.0, 1.0

        x_next = x_func(x_prev, y_prev, a, b, c, d)
        y_next = y_func(x_prev, y_prev, a, b, c, d)

        # Expected: x = sin(1.0*1.0) - cos(1.0*1.0)
        # Expected: y = sin(1.0*1.0) - cos(1.0*1.0)
        assert np.isclose(x_next, np.sin(1.0) - np.cos(1.0))
        assert np.isclose(y_next, np.sin(1.0) - np.cos(1.0))

    def test_johnny_svensson_with_known_values(self):
        """Test Johnny_Svensson attractor with known parameters."""
        x_func, y_func = _get_equation_functions("Johnny_Svensson")

        # Test a single iteration
        x_prev, y_prev = 1.0, 1.0
        a, b, c, d = 1.0, 1.0, 1.0, 1.0

        x_next = x_func(x_prev, y_prev, a, b, c, d)
        y_next = y_func(x_prev, y_prev, a, b, c, d)

        # Expected: x = 1.0*sin(1.0*1.0) - sin(1.0*1.0) = 0
        # Expected: y = 1.0*cos(1.0*1.0) + cos(1.0*1.0) = 2*cos(1.0)
        assert np.isclose(x_next, 0.0)
        assert np.isclose(y_next, 2 * np.cos(1.0))

    def test_custom1_with_known_values(self):
        """Test Custom1 attractor with known parameters."""
        x_func, y_func = _get_equation_functions("Custom1")

        # Test a single iteration
        x_prev, y_prev = 1.0, 1.0
        a, b, c, d = 1.0, 1.0, 1.0, 1.0

        x_next = x_func(x_prev, y_prev, a, b, c, d)
        y_next = y_func(x_prev, y_prev, a, b, c, d)

        # Expected: x = sin(cos(1.0*1.0)) + 1.0*cos(1.0*1.0)
        # Expected: y = sin(cos(1.0*1.0)) + 1.0*cos(1.0*1.0)
        assert np.isclose(x_next, np.sin(np.cos(1.0)) + np.cos(1.0))
        assert np.isclose(y_next, np.sin(np.cos(1.0)) + np.cos(1.0))

    def test_custom2_with_known_values(self):
        """Test Custom2 attractor with known parameters."""
        x_func, y_func = _get_equation_functions("Custom2")

        # Test a single iteration
        x_prev, y_prev = 1.0, 1.0
        a, b, c, d = 1.0, 1.0, 1.0, 1.0

        x_next = x_func(x_prev, y_prev, a, b, c, d)
        y_next = y_func(x_prev, y_prev, a, b, c, d)

        # Expected: x = 1.0*(exp(cos(1.0)) - π/2) + 1.0*(exp(sin(1.0)) - π/2)
        # Expected: y = 1.0*(exp(sin(1.0)) - π/2) + 1.0*(exp(cos(1.0)) - π/2)
        expected_x = (np.exp(np.cos(1.0)) - np.pi / 2) + (
            np.exp(np.sin(1.0)) - np.pi / 2
        )
        expected_y = (np.exp(np.sin(1.0)) - np.pi / 2) + (
            np.exp(np.cos(1.0)) - np.pi / 2
        )
        assert np.isclose(x_next, expected_x)
        assert np.isclose(y_next, expected_y)

    def test_custom3_with_known_values(self):
        """Test Custom3 attractor with known parameters."""
        x_func, y_func = _get_equation_functions("Custom3")

        # Test a single iteration
        x_prev, y_prev = 1.0, 1.0
        a, b, c, d = 1.0, 1.0, 1.0, 1.0

        x_next = x_func(x_prev, y_prev, a, b, c, d)
        y_next = y_func(x_prev, y_prev, a, b, c, d)

        # Expected: x = 1.0*exp(arcsinh(1.0)) - 1.0*exp(sin(1.0))
        # Expected: y = 1.0*exp(arcsinh(1.0)) - 1.0*exp(sin(1.0))
        expected = np.exp(np.arcsinh(1.0)) - np.exp(np.sin(1.0))
        assert np.isclose(x_next, expected)
        assert np.isclose(y_next, expected)

    def test_custom4_with_known_values(self):
        """Test Custom4 attractor with known parameters."""
        x_func, y_func = _get_equation_functions("Custom4")

        # Test a single iteration
        x_prev, y_prev = 1.0, 1.0
        a, b, c, d = 1.0, 1.0, 1.0, 1.0

        x_next = x_func(x_prev, y_prev, a, b, c, d)
        y_next = y_func(x_prev, y_prev, a, b, c, d)

        # Expected: x = 1^2 - 1^2 + 1.0*sin(1.0) + 1.0*sin(1.0*1.0) = 0 + 2*sin(1.0)
        # Expected: y = 1.0*1.0*1.0 + 1.0*1.0 + 1.0*sin(1.0) = 1 + 1 + sin(1.0)
        assert np.isclose(x_next, 2 * np.sin(1.0))
        assert np.isclose(y_next, 2.0 + np.sin(1.0))


class TestEquationSymmetry:
    """Tests for mathematical properties of equations."""

    @pytest.mark.parametrize("equation_id", list(EQUATION_LIBRARY.keys()))
    def test_equations_are_deterministic(self, equation_id):
        """Test that equations produce consistent results."""
        x_func, y_func = _get_equation_functions(equation_id)

        x_prev, y_prev = 0.5, 0.5
        a, b, c, d = 1.0, 1.0, 1.0, 1.0

        # Execute twice with same inputs
        x1 = x_func(x_prev, y_prev, a, b, c, d)
        y1 = y_func(x_prev, y_prev, a, b, c, d)

        x2 = x_func(x_prev, y_prev, a, b, c, d)
        y2 = y_func(x_prev, y_prev, a, b, c, d)

        # Should get identical results
        assert x1 == x2
        assert y1 == y2
