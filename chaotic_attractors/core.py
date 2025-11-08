"""
4 Parameter Chaotic Attractor Visualizer

This program generates and visualizes 4-parameter chaotic attractors.

Users can specify different chaotic attractor equations and parameters
to explore different attractor behaviors.

Author: Alex Spigler
"""

from typing import Tuple, Dict, Any, List, Optional, Callable
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde
import os
import re

from .equations import EQUATION_LIBRARY

# ===================================================
# Configuration Parameters
# ===================================================

# Iteration settings
ITERATIONS_DEFAULT: int = 2_000_000
X_START: float = -0.72
Y_START: float = -0.64

# Visualization settings
ALPHA_DEFAULT: float = 0.3  # 0.0 - 1.0 (fully opaque)
POINT_SIZE_DEFAULT: float = 0.15

# Color configuration
COLOR_METHOD: str = "gradient"  # Options: "viridis", "gradient", "gradient3", "gradientn"

# Viridis palettes
VIRIDIS_PALETTE: str = "plasma"

# Gradient settings
GRADIENT_LOW: str = "lightblue"
GRADIENT_HIGH: str = "darkviolet"

# Gradient3 settings
GRADIENT3_LOW: str = "lightblue"
GRADIENT3_MID: str = "darkviolet"
GRADIENT3_HIGH: str = "blue"
GRADIENT3_MIDPOINT: float = 0.4

# GradientN settings (make sure first value is 0, and last value is 1)
GRADIENTN_COLORS: List[str] = ["aliceblue", "lightblue", "darkviolet", "purple"]
GRADIENTN_VALUES: List[float] = [0, 0.2, 0.65, 1]


# ===================================================
# Equation Compilation
# ===================================================

def _compile_equation(
    eq_str: str, 
    param_names: Tuple[str, ...] = ('a', 'b', 'c', 'd')
) -> Callable:
    """
    Compile equation string into a callable function.
    
    Returns a function that takes (x_prev, y_prev, a, b, c, d) and computes the next value.
    """
    # Replace array notation with previous-value names
    eq_str = eq_str.replace('x[i-1]', 'x_prev')
    eq_str = eq_str.replace('y[i-1]', 'y_prev')
    
    func_str = f"lambda x_prev, y_prev, a, b, c, d: {eq_str}"
    namespace = {'np': np, '__builtins__': {}}

    try:
        return eval(func_str, namespace)
    except Exception as e:
        raise ValueError(f"Failed to compile equation '{eq_str}': {e}")
        
        

def _get_equation_functions(equation_id: str) -> Tuple[Callable, Callable]:
    """Get compiled (x_update, y_update) functions for the specified equation."""
    if equation_id not in EQUATION_LIBRARY:
        raise KeyError(
            f"Unknown equation: '{equation_id}'. "
        )
    
    eq_dict = EQUATION_LIBRARY[equation_id]
    x_func = _compile_equation(eq_dict['x_eq'])
    y_func = _compile_equation(eq_dict['y_eq'])
    
    return x_func, y_func


# ===================================================
# Core Functions
# ===================================================

def generate_chaotic(
    params: Dict[str, float], 
    equation_id: str, 
    n_iterations: int = ITERATIONS_DEFAULT,
    x_start: float = X_START,
    y_start: float = Y_START
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Generate trajectory points for a chaotic attractor using iterative equations.
    
    Example:
        >>> params = {'a': 0.9, 'b': -0.6, 'c': 2.0, 'd': 0.5}
        >>> x, y = generate_chaotic(params, "Tinkerbell", n_iterations=10000)
    """
    # Validate parameters
    required_params = {'a', 'b', 'c', 'd'}
    if not required_params.issubset(params.keys()):
        missing = required_params - params.keys()
        raise KeyError(f"Missing required parameters: {missing}")
    
    a, b, c, d = params['a'], params['b'], params['c'], params['d']
    
    # Pre-allocate arrays
    x = np.zeros(n_iterations, np.float64)
    y = np.zeros(n_iterations, np.float64)
    
    print(f"Generating attractor: {equation_id}")
    print(f"Parameters: a={params['a']}, b={params['b']}, c={params['c']}, d={params['d']}")
    print(f"Iterations: {n_iterations:,}")
    
    # Set initial conditions
    x[0] = x_start
    y[0] = y_start
    
    # Get compiled equation functions
    x_func, y_func = _get_equation_functions(equation_id)
    
    # Iterate through the dynamical system
    valid_length = n_iterations
    for i in range(1, n_iterations):
        try:
            x[i] = x_func(x[i-1], y[i-1], a, b, c, d)
            y[i] = y_func(x[i-1], y[i-1], a, b, c, d)
            
            # Early termination if values become non-finite
            if not (np.isfinite(x[i]) and np.isfinite(y[i])):
                valid_length = i
                break
                
        except (FloatingPointError, OverflowError):
            valid_length = i
            break
        
    x = x[:valid_length]
    y = y[:valid_length]
    
    return x, y


def prepare_attractor_data(
    params: Dict[str, float], 
    equation_id: str, 
    n_iterations: int = ITERATIONS_DEFAULT,
    kde_sample_size: int = 50_000
) -> Dict[str, Any]:
    """
    Generate attractor points and compute density using KDE for visualization.
    
    Returns dict with keys: 'x', 'y', 'density' (normalized 0-1), 'params'.
    Raises ValueError if fewer than 10,000 valid points are generated.
    """
    # Generate points
    x, y = generate_chaotic(params, equation_id, n_iterations)
    
    print(f"Generated {len(x):,} valid points")
    print(f"X range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Y range: [{y.min():.3f}, {y.max():.3f}]")

    # Remove any NaN or Inf values
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    # Validate sufficient data
    if len(x) < 10_000:
        raise ValueError(
            f"Insufficient valid points for plotting an attractor:\n"
            f"Generated {len(x)}, need at least 10,000."
        )
    
    # Compute density using KDE
    try:
        # Sample for KDE efficiency
        sample_size = min(len(x), kde_sample_size)
        indices = np.random.choice(len(x), sample_size, replace=False)
        x_sample = x[indices]
        y_sample = y[indices]
        
        # Calculate KDE
        kde = gaussian_kde(np.vstack([x_sample, y_sample]))
        density = kde(np.vstack([x, y]))
        
        print(f"Initial Density Range: {density.min():.4f} to {density.max():.4f}")
        
        # Normalize density to [0, 1]
        density_range = density.max() - density.min()
        if density_range != 0:
            density = (density - density.min()) / density_range
            print(f"Normalized Density Range: {density.min():.4f} to {density.max():.4f}")
        else:
            density = np.ones(len(x), np.float64)
            print("Warning: Density is a constant--normalized to 1")
        
    except Exception as e:
        print(f"Warning: Could not compute KDE density: {e}")
        print("Falling back to uniform density")
        density = np.ones(len(x), np.float64)
    
    return {
        'x': x,
        'y': y,
        'density': density,
        'params': params
    }


def create_colormap(
    method: str, 
    **kwargs: Any
) -> LinearSegmentedColormap:
    """
    Create colormap for attractor visualization.
    
    Methods: "viridis", "gradient", "gradient3", "gradientn".
    Pass method-specific colors/settings via kwargs.
    """
    if method == "viridis":
        palette = kwargs.get('palette', VIRIDIS_PALETTE)
        return plt.cm.get_cmap(palette)
    
    elif method == "gradient":
        low = kwargs.get('low', GRADIENT_LOW)
        high = kwargs.get('high', GRADIENT_HIGH)
        return LinearSegmentedColormap.from_list('custom', [low, high])
    
    elif method == "gradient3":
        low = kwargs.get('low', GRADIENT3_LOW)
        mid = kwargs.get('mid', GRADIENT3_MID)
        high = kwargs.get('high', GRADIENT3_HIGH)
        midpoint = kwargs.get('midpoint', GRADIENT3_MIDPOINT)
        colors = [low, mid, high]
        positions = [0, midpoint, 1]
        return LinearSegmentedColormap.from_list('custom', list(zip(positions, colors)))
    
    elif method == "gradientn":
        colors = kwargs.get('colors', GRADIENTN_COLORS)
        values = kwargs.get('values', GRADIENTN_VALUES)
        return LinearSegmentedColormap.from_list('custom', list(zip(values, colors)))
    
    else:
        print(f"Warning: Unknown color method '{method}', defaulting to viridis")
        return plt.cm.viridis


def plot_chaotic(
    data: Dict[str, Any], 
    point_size: float = POINT_SIZE_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    background_color: str = 'white',
    color_method: str = COLOR_METHOD,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    **color_kwargs: Any
) -> Tuple[Figure, Axes]:
    """Create and return matplotlib figure and axes with the attractor plotted."""
    x = data['x']
    y = data['y']
    density = data['density']
    
    # Calculate aspect ratio
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    
    if y_range == 0 or x_range == 0:
        aspect_ratio = 1.0
    else:
        aspect_ratio = x_range / y_range
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        width = 12
        height = width / aspect_ratio
        figsize = (width, height)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=background_color)
    
    # Create colormap
    cmap = create_colormap(color_method, **color_kwargs)
    
    # Plot the attractor
    ax.scatter(
        x, y,
        c=density,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        edgecolors='none',
        rasterized=False
    )
    
    # Formatting
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(background_color)
    
    return fig, ax


def save_attractor(
    data: Dict[str, Any],
    filename: str,
    point_size: float = POINT_SIZE_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    save_format: str = 'png',
    include_info: bool = True,
    equation_id: Optional[str] = None,
    **plot_kwargs: Any
) -> List[str]:
    """
    Generate and save attractor visualization to file(s).
    
    Set save_format to 'all' for PNG, PDF, and SVG. 
    If include_info=True, equation_id is required.
    Returns list of saved file paths.
    """
    # Validate arguments
    if include_info and equation_id is None:
        raise ValueError("equation_id is required when include_info=True")
    
    # Remove extension if provided
    base_filename = os.path.splitext(filename)[0]
    
    # Determine which formats to save
    if save_format == 'all':
        formats = ['png', 'pdf', 'svg']
    else:
        formats = [save_format]
    
    saved_files = []
    
    for fmt in formats:
        output_file = f"{base_filename}.{fmt}"
        
        if include_info and equation_id:
            # Create figure with info panel
            fig, saved_file = create_attractor_with_eq(
                data, output_file, equation_id,
                point_size, alpha, **plot_kwargs
            )
        else:
            # Standard attractor without info
            fig, ax = plot_chaotic(
                data, 
                point_size=point_size, 
                alpha=alpha, 
                **plot_kwargs
            )
            
            # Format-specific settings
            save_kwargs = {
                'bbox_inches': 'tight',
                'pad_inches': 0.05,
                'facecolor': fig.get_facecolor()
            }
            
            if fmt in ('png', 'pdf'):
                save_kwargs['dpi'] = fig.dpi
            
            plt.savefig(output_file, **save_kwargs)
            plt.close(fig)
        
        saved_files.append(output_file)
        print(f"Saved: {output_file}")
    
    return saved_files


def convert_to_math_text(eq_str: str) -> str:
    """Convert Python equation syntax to matplotlib math text for matplotlib rendering."""
    # Remove Python-specific syntax
    eq_str = eq_str.replace('np.', '')
    eq_str = eq_str.replace('[i-1]', '_i')  # Array index to subscript
    
    # Convert operators to LaTeX
    eq_str = eq_str.replace(' * ', r' \cdot ')
    eq_str = eq_str.replace('pi', r'\pi')
    
    # Convert functions to LaTeX commands (order matters!)
    function_map = {
        'arcsinh': r'\mathrm{arcsinh}',
        'arccosh': r'\mathrm{arccosh}',
        'arctanh': r'\mathrm{arctanh}',
        'arcsin': r'\arcsin',
        'arccos': r'\arccos',
        'arctan': r'\arctan',
        'sinh': r'\sinh',
        'cosh': r'\cosh',
        'tanh': r'\tanh',
        'sin': r'\sin',
        'cos': r'\cos',
        'tan': r'\tan',
        'exp': r'\exp',
    }
    
    for func, latex_func in function_map.items():
        eq_str = eq_str.replace(f'{func}(', f'{latex_func}(')
    
    # Convert exponentiation: **2 -> ^{2}
    eq_str = re.sub(r'\*\*(\d+)', r'^{\1}', eq_str)
    
    return eq_str


def create_attractor_with_eq(
    data: Dict[str, Any],
    output_file: str,
    equation_id: str,
    point_size: float = POINT_SIZE_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    **plot_kwargs: Any
) -> Tuple[Figure, str]:
    """
    Create and save attractor visualizations with equation and parameter info panel below.
    
    Returns tuple of (matplotlib Figure, output file path).
    """
    params = data['params']
    x = data['x']
    y = data['y']
    
    # Calculate aspect ratio
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    
    if y_range == 0 or x_range == 0:
        aspect_ratio = 1.0
    else:
        aspect_ratio = x_range / y_range
    
    # Get equations for display
    x_eq_raw = EQUATION_LIBRARY[equation_id]['x_eq']
    y_eq_raw = EQUATION_LIBRARY[equation_id]['y_eq']

    # Convert to LaTeX math text
    x_eq_math = convert_to_math_text(x_eq_raw)
    y_eq_math = convert_to_math_text(y_eq_raw)
    
    # Create figure with space for text panel below
    figsize = plot_kwargs.get('figsize', None)
    if figsize is None:
        width = 12
        height = width / aspect_ratio
        figsize = (width, height * 1.10)  # Add 10% space for text
    
    dpi = plot_kwargs.get('dpi', 300)
    background_color = plot_kwargs.get('background_color', 'white')
    
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=background_color)
    
    # Create grid: main plot gets 90%, text panel gets 10%
    gs = fig.add_gridspec(
        2, 1, 
        height_ratios=[9, 1],
        hspace=0.05,
        left=0.05,
        right=0.95,
        top=0.95,
        bottom=0.05
    )
    
    # Main attractor plot
    ax_main = fig.add_subplot(gs[0, 0])
    
    # Get colormap
    color_method = plot_kwargs.get('color_method', COLOR_METHOD)
    cmap = create_colormap(color_method, **plot_kwargs)
    
    # Plot attractor
    ax_main.scatter(
        x, y,
        c=data['density'],
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        edgecolors='none',
        rasterized=False
    )
    
    ax_main.set_aspect('equal')
    ax_main.axis('off')
    ax_main.set_facecolor(background_color)
    
    # Text panel for equations and parameters
    ax_text = fig.add_subplot(gs[1, 0])
    ax_text.axis('off')
    ax_text.set_facecolor(background_color)
    
    # Build info text
    info_lines = [
        rf"$x_{{i+1}} = {x_eq_math}$",
        rf"$y_{{i+1}} = {y_eq_math}$",
        "",  # Blank line for spacing
        f"$a = {params['a']},  b = {params['b']},  c = {params['c']},  d = {params['d']}$",
        rf"$x_0 = {X_START},  y_0 = {Y_START}$"
    ]
    info_text = '\n'.join(info_lines)

    # Add centered text to panel
    ax_text.text(
        0.5, 0.5, info_text,
        ha='center',
        va='center',
        fontsize=10,
        transform=ax_text.transAxes
    )
    
    # Save with format-specific settings
    fmt = os.path.splitext(output_file)[1][1:]  # Get extension without dot
    save_kwargs = {
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
        'facecolor': fig.get_facecolor()
    }
    
    if fmt in ('png', 'pdf'):
        save_kwargs['dpi'] = dpi
    
    print("\nSaving visualization(s)...")
    plt.savefig(output_file, **save_kwargs)
    plt.close(fig)
    
    print("\nDone!")
    
    return fig, output_file
