"""
Unit tests for core attractor generation and visualization functions.
"""

import pytest
import numpy as np
from chaotic_attractors.core import (
    generate_chaotic,
    prepare_generate_data,
    _compile_equation,
    _get_equation_functions,
    create_colormap,
    convert_to_math_text,
)


class TestGenerateChaotic:
    """Tests for the generate_chaotic function."""
    
    def test_returns_correct_types(self):
        """Test that generate_chaotic returns numpy arrays."""
        params = {'a': 0.9, 'b': -0.6013, 'c': 2.0, 'd': 0.5}
        x_start=-0.72
        y_start=-0.64
        x, y = generate_chaotic(params=params, x_start=x_start,
                                y_start=y_start, equation_id="Tinkerbell",
                                iterations=1000)
        
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.dtype == np.float64
        assert y.dtype == np.float64
    
    def test_returns_equal_length_arrays(self):
        """Test that x and y arrays have the same length."""
        params = {'a': 0.9, 'b': -0.6013, 'c': 2.0, 'd': 0.5}
        x_start=-0.72
        y_start=-0.64
        x, y = generate_chaotic(params=params, x_start=x_start,
                                y_start=y_start, equation_id="Tinkerbell",
                                iterations=1000)
        
        assert len(x) == len(y)
    
    def test_respects_iteration_limit(self):
        """Test that output length doesn't exceed requested iterations."""
        params = {'a': 0.9, 'b': -0.6013, 'c': 2.0, 'd': 0.5}
        iterations = 5000
        x_start=-0.72
        y_start=-0.64
        x, y = generate_chaotic(params=params, x_start=x_start,
                                y_start=y_start, equation_id="Tinkerbell",
                                iterations=iterations)
        
        assert len(x) <= iterations
        assert len(y) <= iterations
    
    def test_handles_divergent_parameters(self):
        """Test that divergent parameters terminate early without crashing."""
        params = {'a': 100.0, 'b': 100.0, 'c': 100.0, 'd': 100.0}
        x_start=-0.72
        y_start=-0.64
        x, y = generate_chaotic(params=params, x_start=x_start,
                                y_start=y_start, equation_id="Tinkerbell",
                                iterations=1000)
        
        # Should terminate early due to non-finite values
        assert len(x) < 10000
    
    def test_initial_conditions_are_set(self):
        """Test that initial conditions are properly set."""
        params = {'a': 0.9, 'b': -0.6013, 'c': 2.0, 'd': 0.5}
        x, y = generate_chaotic(
            params=params, equation_id="Tinkerbell", 
            iterations=100,
            x_start=-0.72,
            y_start=-0.64
        )
        
        assert x[0] == -0.72
        assert y[0] == -0.64
    
    def test_raises_error_for_missing_parameters(self):
        """Test that missing parameters raise KeyError."""
        params = {'a': 0.9, 'b': -0.6013}  # Missing c and d
        x_start=-0.72
        y_start=-0.64
        
        with pytest.raises(KeyError, match="Missing required parameters"):
            x, y = generate_chaotic(params=params, x_start=x_start,
                                    y_start=y_start, equation_id="Tinkerbell",
                                    iterations=1000)
    
    def test_raises_error_for_unknown_equation(self):
        """Test that unknown equation IDs raise KeyError."""
        params = {'a': 0.9, 'b': -0.6013, 'c': 2.0, 'd': 0.5}
        x_start=-0.72
        y_start=-0.64
        
        with pytest.raises(KeyError, match="Unknown equation"):
            x, y = generate_chaotic(params=params, x_start=x_start,
                                    y_start=y_start, equation_id="Unknown",
                                    iterations=1000)


class TestPrepareGenerateData:
    """Tests for the prepare_attractor_data function."""
    
    def test_returns_dict_with_required_keys(self):
        """Test that output dictionary contains all required keys."""
        params = {'a': 0.9, 'b': -0.6013, 'c': 2.0, 'd': 0.5}
        x_start=-0.72
        y_start=-0.64
        data = prepare_generate_data(params=params, equation_id="Tinkerbell",
                                     x_start=x_start, y_start=y_start, final_iterations=50000)
        
        assert 'x' in data
        assert 'y' in data
        assert 'density' in data
        assert 'params' in data
    
    def test_density_normalized_to_zero_one(self):
        """Test that density values are normalized to [0, 1] range."""
        params = {'a': 0.9, 'b': -0.6013, 'c': 2.0, 'd': 0.5}
        x_start=-0.72
        y_start=-0.64
        data = prepare_generate_data(params=params, equation_id="Tinkerbell",
                                     x_start=x_start, y_start=y_start, final_iterations=50000)
        
        assert data['density'].min() == 0.0
        assert data['density'].max() == 1.0
    
    def test_raises_error_for_insufficient_points(self):
        """Test that insufficient valid points raise ValueError."""
        params = {'a': 100.0, 'b': 100.0, 'c': 100.0, 'd': 100.0}
        x_start=-0.72
        y_start=-0.64
        
        with pytest.raises(ValueError, match="Insufficient valid points"):
            prepare_generate_data(params=params, equation_id="Tinkerbell",
                                  x_start=x_start, y_start=y_start, final_iterations=5000)


class TestEquationCompilation:
    """Tests for equation compilation and retrieval."""
    
    def test_compile_simple_equation(self):
        """Test compilation of a simple equation."""
        eq_str = "a * x[n-1] + b * y[n-1]"
        func = _compile_equation(eq_str)
        
        # Test the compiled function
        result = func(1.0, 2.0, 0.5, 0.3, 0.0, 0.0)
        expected = 0.5 * 1.0 + 0.3 * 2.0
        assert np.isclose(result, expected)
    
    def test_compile_equation_with_numpy_functions(self):
        """Test compilation with numpy functions like sin and cos."""
        eq_str = "np.sin(a * x[n-1]) + np.cos(b * y[n-1])"
        func = _compile_equation(eq_str)
        
        result = func(0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
        expected = np.sin(0.0) + np.cos(0.0)
        assert np.isclose(result, expected)
    
    def test_get_equation_functions_returns_two_functions(self):
        """Test that _get_equation_functions returns x and y update functions."""
        x_func, y_func = _get_equation_functions("Clifford")
        
        assert callable(x_func)
        assert callable(y_func)
    
    def test_raises_error_for_invalid_equation_syntax(self):
        """Test that invalid equation syntax raises ValueError."""
        eq_str = "this is not valid python"
        
        with pytest.raises(ValueError, match="Failed to compile equation"):
            _compile_equation(eq_str)


class TestColormap:
    """Tests for colormap creation."""
    
    def test_viridis_method(self):
        """Test that viridis method returns a valid colormap."""
        cmap = create_colormap("viridis", palette="plasma")
        assert cmap is not None
    
    def test_gradient_method(self):
        """Test that gradient method creates a colormap."""
        cmap = create_colormap("gradient", low="blue", high="red")
        assert cmap is not None
    
    def test_gradient3_method(self):
        """Test that gradient3 method creates a colormap."""
        cmap = create_colormap(
            "gradient3", 
            low="blue", 
            mid="green", 
            high="red",
            midpoint=0.5
        )
        assert cmap is not None
        
    def test_gradientn_method(self):
        """Test that gradientn method creates a colormap."""
        GRADIENTN_COLORS = ["red", "orange", "yellow", "green", "blue"]
        GRADIENTN_VALUES = [0, 0.2, 0.4, 0.6, 1]
        cmap = create_colormap(
            "gradientn",
            colors=GRADIENTN_COLORS,
            values=GRADIENTN_VALUES
        )
        assert cmap is not None
    
    def test_unknown_method_returns_default(self):
        """Test that unknown methods default to viridis."""
        cmap = create_colormap("nonexistent_method")
        assert cmap is not None


class TestConvertToMathText:
    """Tests for Matplotlib conversion."""
    
    def test_removes_numpy_prefix(self):
        """Test that 'np.' prefix is removed."""
        result = convert_to_math_text("np.sin(x)")
        assert "np." not in result
    
    def test_converts_array_notation(self):
        """Test that array notation is converted to subscripts."""
        result = convert_to_math_text("x[n-1]")
        assert "_n" in result
        assert "[n-1]" not in result
    
    def test_converts_multiplication(self):
        """Test that multiplication symbol is removed."""
        result = convert_to_math_text("a * x")
        assert r"\cdot" not in result
    
    def test_converts_exponentiation(self):
        """Test that exponentiation is converted to LaTeX."""
        result = convert_to_math_text("x**2")
        assert "^{2}" in result
        assert "**" not in result
    
    def test_converts_trig_functions(self):
        """Test that trig functions are converted to LaTeX."""
        result = convert_to_math_text("sin(x) + cos(y)")
        assert r"\sin" in result
        assert r"\cos" in result
