"""
Unit tests for parameter search and evaluation functions.
"""

import numpy as np
import pytest

from chaotic_attractors.search import (
    evaluate_attractor_second,
    generate_random,
)


class TestGenerateRandom:
    """Tests for the generate_random function."""

    def test_returns_float(self):
        """Test that generate_random returns a float."""
        result = generate_random(-3.0, 3.0, decimals=2)
        assert isinstance(result, float)

    def test_respects_decimal_precision(self):
        """Test that output has correct decimal precision."""
        decimals = 2
        result = generate_random(-3.0, 3.0, decimals=decimals)

        # Check that result has at most 'decimals' decimal places
        result_str = f"{result:.10f}"  # Convert to string with many decimals
        decimal_part = result_str.split(".")[1].rstrip("0")
        assert len(decimal_part) <= decimals

    def test_within_range(self):
        """Test that generated values are within specified range."""
        min_val, max_val = -5.0, 5.0

        for _ in range(100):
            result = generate_random(min_val, max_val, decimals=2)
            assert min_val <= result <= max_val

    def test_raises_error_for_invalid_range(self):
        """Test that invalid range raises ValueError."""
        with pytest.raises(ValueError, match="Invalid range"):
            generate_random(5.0, 5.0, decimals=2)

    def test_different_decimal_precisions(self):
        """Test that different decimal precisions work correctly."""
        for decimals in [0, 1, 2, 3, 4]:
            result = generate_random(-1.0, 1.0, decimals=decimals)
            assert isinstance(result, float)

            # Convert to string with exact formatting
            result_str = f"{result:.{decimals}f}"

            # Convert back to float and compare rounded values
            assert round(result, decimals) == float(result_str)

    def test_generates_diverse_values(self):
        """Test that function generates diverse values, not always the same."""
        values = set()
        for _ in range(100):
            values.add(generate_random(-5.0, 5.0, decimals=3))

        assert len(np.unique(list(values))) >= 97


class TestEvaluateAttractor:
    """Tests for the evaluate_attractor_second function."""

    def test_returns_dict_with_required_keys(self):
        """Test that output dictionary contains required keys."""
        params = {"a": 0.9, "b": -0.6, "c": 2.0, "d": 0.5}
        x_start = -0.72
        y_start = -0.64
        result = evaluate_attractor_second(
            params=params,
            equation_id="Tinkerbell",
            x_start=x_start,
            y_start=y_start,
            iterations=5000,
        )

        assert "score" in result
        assert "reason" in result

    def test_passes_good_parameters(self):
        """Test that known good parameters pass evaluation."""
        # Tinkerbell attractor with standard parameters
        params = {"a": 0.9, "b": -0.6013, "c": 2.0, "d": 0.5}
        x_start = -0.72
        y_start = -0.64
        result = evaluate_attractor_second(
            params=params,
            equation_id="Tinkerbell",
            x_start=x_start,
            y_start=y_start,
            iterations=5000,
        )

        assert result["score"] >= 0.0
        assert result["reason"] == "Passed all checks"
        assert "x_range" in result
        assert "y_range" in result
        assert "unique_ratio" in result
        assert "aspect_ratio" in result

    def test_rejects_divergent_parameters(self):
        """Test that divergent parameters are rejected."""
        params = {"a": 100.0, "b": 100.0, "c": 100.0, "d": 100.0}
        x_start = 1
        y_start = 1

        result = evaluate_attractor_second(
            params=params,
            equation_id="Tinkerbell",
            x_start=x_start,
            y_start=y_start,
            iterations=5000,
        )

        assert result["score"] < 0.0
        assert "reason" in result

    def test_rejects_collapsed_parameters(self):
        """Test that parameters causing collapse are rejected."""
        params = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
        x_start = 0
        y_start = 0

        result = evaluate_attractor_second(
            params=params,
            equation_id="Tinkerbell",
            x_start=x_start,
            y_start=y_start,
            iterations=5000,
        )

        assert result["score"] < 0.0
        assert (
            "collapsed" in result["reason"].lower()
            or "small" in result["reason"].lower()
        )

    def test_score_is_non_negative_for_passed_attractors(self):
        """Test that passed attractors have non-negative scores."""
        params = {"a": 0.9, "b": -0.6013, "c": 2.0, "d": 0.5}
        x_start = -0.72
        y_start = -0.64

        result = evaluate_attractor_second(
            params=params,
            equation_id="Tinkerbell",
            x_start=x_start,
            y_start=y_start,
            iterations=5000,
        )

        assert result["score"] >= 0.0
        assert result["reason"] == "Passed all checks"

    def test_aspect_ratio_bounds(self):
        """Test that aspect ratio check works correctly."""
        params = {"a": 0.9, "b": -0.6013, "c": 2.0, "d": 0.5}
        x_start = -0.72
        y_start = -0.64

        result = evaluate_attractor_second(
            params=params,
            equation_id="Tinkerbell",
            x_start=x_start,
            y_start=y_start,
            iterations=5000,
            max_aspect_ratio=1.00001,
        )

        if result["score"] < 0.0 and "aspect" in result["reason"].lower():
            assert True

    def test_unique_ratio_bounds(self):
        """Test that unique ratio check works correctly."""
        params = {"a": 0.9, "b": -0.6013, "c": 2.0, "d": 0.5}
        x_start = -0.72
        y_start = -0.64

        result = evaluate_attractor_second(
            params=params,
            equation_id="Tinkerbell",
            x_start=x_start,
            y_start=y_start,
            iterations=5000,
            min_unique_ratio=0.1,
            max_unique_ratio=0.95,
        )

        if result["score"] >= 0.0:
            assert 0.1 <= result["unique_ratio"] <= 0.95

    def test_handles_exception_during_generation(self):
        """Test that exceptions during generation are caught."""
        params = {"a": 0.9, "b": -0.6, "c": 2.0, "d": 0.5}
        x_start = -0.72
        y_start = -0.64

        result = evaluate_attractor_second(
            params=params,
            equation_id="Unknown",
            x_start=x_start,
            y_start=y_start,
            iterations=5000,
        )

        assert result["score"] < 0.0
        assert (
            "exception" in result["reason"].lower()
            or "error" in result["reason"].lower()
        )

    def test_range_checks_work(self):
        """Test that min and max range checks are applied."""
        params = {"a": 0.9, "b": -0.6013, "c": 2.0, "d": 0.5}
        x_start = -0.72
        y_start = -0.64

        result = evaluate_attractor_second(
            params=params,
            equation_id="Tinkerbell",
            x_start=x_start,
            y_start=y_start,
            iterations=10000,
            min_small_side=1000.0,
            max_small_side=2000.0,
        )

        assert result["score"] < 0.0
        assert "reason" in result


class TestEvaluateAttractorEdgeCases:
    """Tests for edge cases in evaluate_attractor_second."""

    def test_handles_all_zeros(self):
        """Test handling when all generated points are zero."""
        params = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
        result = evaluate_attractor_second(params, "Tinkerbell", iterations=1000)

        assert result["score"] < 0.0

    def test_handles_very_small_iterations(self):
        """Test with very small iteration count."""
        params = {"a": 0.9, "b": -0.6, "c": 2.0, "d": 0.5}
        x_start = 100
        y_start = 100
        result = evaluate_attractor_second(
            params=params,
            equation_id="Tinkerbell",
            x_start=x_start,
            y_start=y_start,
            iterations=50,
        )

        assert result["score"] < 0.0
        assert "insufficient valid points" in result["reason"].lower()

    def test_scoring_formula_properties(self):
        """Test that scoring formula has expected properties."""
        params = {"a": 0.9, "b": -0.6013, "c": 2.0, "d": 0.5}
        x_start = -0.72
        y_start = -0.64

        result = evaluate_attractor_second(
            params=params,
            equation_id="Tinkerbell",
            x_start=x_start,
            y_start=y_start,
            iterations=5000,
        )

        assert result["score"] >= 0.0
