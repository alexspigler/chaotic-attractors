"""
4 Parameter Chaotic Attractor Visualizer

This program generates and visualizes 4-parameter chaotic attractors.

Users can specify different chaotic attractor equations and parameters
to explore different attractor behaviors.

Author: Alex Spigler
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import os
import re

# ===================================================
# Configuration Parameters
# ===================================================

# Iteration settings
ITERATIONS_DEFAULT = 2_000_000
X_START = -0.72
Y_START = -0.64

# Visualization settings
ALPHA_DEFAULT = 0.3  # 0.0 - 1.0 (fully opaque)
POINT_SIZE_DEFAULT = 0.15

# Color configuration
COLOR_METHOD = "gradient"  # Options: "viridis", "gradient", "gradient3", "gradientn"

# Viridis palettes
VIRIDIS_PALETTE = "plasma"

# Gradient settings
GRADIENT_LOW = "lightblue"
GRADIENT_HIGH = "darkviolet"

# Gradient3 settings
GRADIENT3_LOW = "lightblue"
GRADIENT3_MID = "darkviolet"
GRADIENT3_HIGH = "blue"
GRADIENT3_MIDPOINT = 0.4

# GradientN settings (make sure first value is 0, and last value is 1)
GRADIENTN_COLORS = ["aliceblue", "lightblue", "darkviolet", "purple"]
GRADIENTN_VALUES = [0, 0.2, 0.65, 1]


# ===================================================
# Equation Library
# ===================================================

EQUATION_LIBRARY = {
    # Clifford
    "Clifford": {
        "x_eq": "np.sin(a * y[i-1]) + c * np.cos(a * x[i-1])",
        "y_eq": "np.sin(b * x[i-1]) + d * np.cos(b * y[i-1])"
    },
    
    # Tinkerbell (x_0, y_0 != 0, 0)
    "Tinkerbell": {
        "x_eq": "x[i-1]**2 - y[i-1]**2 + a * x[i-1] + b * y[i-1]",
        "y_eq": "2 * x[i-1] * y[i-1] + c * x[i-1] + d * y[i-1]"
    },
    
    # Fractal Dreams
    "Fractal_Dreams": {
        "x_eq": "np.sin(b * y[i-1]) + c * np.sin(b * x[i-1])",
        "y_eq": "np.sin(a * x[i-1]) + d * np.sin(a * y[i-1])"
    },
    
    # Peter de Jong
    "Peter_Jong": {
        "x_eq": "np.sin(a * y[i-1]) - np.cos(b * x[i-1])",
        "y_eq": "np.sin(c * x[i-1]) - np.cos(d * y[i-1])"
    },
    
    # Johnny Svensson
    "Johnny_Svensson": {
        "x_eq": "d * np.sin(a * x[i-1]) - np.sin(b * y[i-1])",
        "y_eq": "c * np.cos(a * x[i-1]) + np.cos(b * y[i-1])"
    },
    
    # Custom 1
    "Custom1": {
        "x_eq": "np.sin(np.cos(a * y[i-1])) + c * np.cos(a * x[i-1])",
        "y_eq": "np.sin(np.cos(b * x[i-1])) + d * np.cos(b * y[i-1])"
    },
    
    # Custom 2
    "Custom2": {
        "x_eq": "a * (np.exp(np.cos(x[i-1])) - np.pi / 2) + b * (np.exp(np.sin(y[i-1])) - np.pi / 2)",
        "y_eq": "c * (np.exp(np.sin(x[i-1])) - np.pi / 2) + d * (np.exp(np.cos(y[i-1])) - np.pi / 2)"
    },
    
    # Custom 3
    "Custom3": {
        "x_eq": "a * np.exp(np.arcsinh(x[i-1])) - b * np.exp(np.sin(y[i-1]))",
        "y_eq": "c * np.exp(np.arcsinh(y[i-1])) - d * np.exp(np.sin(x[i-1]))"
    },
    
    # Custom 4 (x_0, y_0 != 0, 0)
    "Custom4": {
        "x_eq": "x[i-1]**2 - y[i-1]**2 + a * np.sin(x[i-1]) + b * np.sin(b * y[i-1])",
        "y_eq": "a * x[i-1] * y[i-1] + c * x[i-1] + d * np.sin(y[i-1])"
    }
}


# ===================================================
# Core Functions
# ===================================================

def generate_chaotic(params, equation_id, n_iterations=ITERATIONS_DEFAULT):
    """
    Generate chaotic attractor points
    
    Args:
        params: dict with keys 'a', 'b', 'c', 'd'
        equation_id: str, equation name from EQUATION_LIBRARY
        n_iterations: int, number of iterations
        
    Returns:
        tuple: (x_array, y_array)
    """
    a, b, c, d = params['a'], params['b'], params['c'], params['d']
    
    x = np.zeros(n_iterations)
    y = np.zeros(n_iterations)
    
    # Starting point
    x[0] = X_START
    y[0] = Y_START
    
    selected_eq = EQUATION_LIBRARY[equation_id]
    x_expr = selected_eq['x_eq']
    y_expr = selected_eq['y_eq']
    
    for i in range(1, n_iterations):
        try:
            x[i] = eval(x_expr)
            y[i] = eval(y_expr)
        except FloatingPointError:
            break
    
    return x, y


def prepare_attractor_data(params, equation_id, n_iterations=ITERATIONS_DEFAULT):
    """
    Generate points and compute density for visualization
    
    Args:
        params: dict with keys 'a', 'b', 'c', 'd'
        equation_id: str, equation name
        n_iterations: int, number of iterations
        
    Returns:
        dict: {'x': array, 'y': array, 'density': array, 'params': params}
    """
    # Generate points
    x, y = generate_chaotic(params, equation_id, n_iterations)
    
    # Remove any NaN or Inf values
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    if len(x) < 10_000:
        raise ValueError("Not enough valid points generated")
    
    # Compute density using KDE
    try:
        # Sample for KDE
        sample_size = min(len(x), 50000)
        indices = np.random.choice(len(x), sample_size, replace=False)
        x_sample = x[indices]
        y_sample = y[indices]
        
        # Calculate KDE
        kde = gaussian_kde(np.vstack([x_sample, y_sample]))
        density = kde(np.vstack([x, y]))
        
        print(f"Initial Density Range: {density.min():.4f} to {density.max():.4f}")
        
        # Normalize density
        density = (density - density.min()) / (density.max() - density.min())
        
        print(f"Normalized Density Range: {density.min():.4f} to {density.max():.4f}")
        
    except Exception as e:
        print(f"Warning: Could not compute KDE density: {e}")
        density = np.ones(len(x))
    
    return {
        'x': x,
        'y': y,
        'density': density,
        'params': params
    }


def create_colormap(method, **kwargs):
    """Create colormap based on method"""
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
        return plt.cm.viridis


def plot_chaotic(data, 
                 point_size=POINT_SIZE_DEFAULT,
                 alpha=ALPHA_DEFAULT,
                 background_color='white',
                 color_method=COLOR_METHOD,
                 figsize=None,
                 dpi=300,
                 **color_kwargs):
    """
    Plot the chaotic attractor
    
    Args:
        data: dict from prepare_attractor_data
        point_size: float, point size
        alpha: float, transparency
        background_color: str, background color
        color_method: str, coloring method
        figsize: tuple, figure size (width, height)
        dpi: int, resolution
        **color_kwargs: additional color parameters
        
    Returns:
        matplotlib figure and axes
    """
    x = data['x']
    y = data['y']
    density = data['density']
    
    # Calculate appropriate figure size if not provided
    if figsize is None:
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        aspect = x_range / y_range
        width = 12
        height = width / aspect
        figsize = (width, height)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    
    # Get colormap
    cmap = create_colormap(color_method, **color_kwargs)
    
    # Plot points
    ax.scatter(x, y, 
               c=density, 
               cmap=cmap,
               s=point_size, 
               alpha=alpha,
               edgecolors='none',
               rasterized=False)
    
    # Remove axes
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Tight layout
    plt.tight_layout(pad=0)
    
    return fig, ax


def save_attractor(data,
                  filename, 
                  point_size=POINT_SIZE_DEFAULT,
                  alpha=ALPHA_DEFAULT,
                  save_format='pdf',
                  include_info=True,
                  equation_id=None,
                  **plot_kwargs):
    """
    Generate and save attractor visualization
    
    Args:
        data: dict from prepare_attractor_data
        filename: str, output filename (without extension, will be added based on format)
        point_size: float, marker size
        alpha: float, transparency
        save_format: str, output format: 'png', 'pdf', 'svg', or 'all'
        include_info: bool, if True, add text panel with equations and parameters
        equation_id: str, equation name (required if include_info=True)
        **plot_kwargs: additional plotting parameters
    
    Returns:
        list: paths of saved files
    """
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
                point_size, alpha, **plot_kwargs)
        else:
            # Standard attractor without info
            fig, ax = plot_chaotic(data, point_size=point_size, alpha=alpha, **plot_kwargs)
            
            # Format-specific settings
            save_kwargs = {
                'bbox_inches': 'tight',
                'pad_inches': 0.05,
                'facecolor': fig.get_facecolor()
            }
            
            if fmt == 'png':
                save_kwargs['dpi'] = fig.dpi
            elif fmt == 'pdf':
                save_kwargs['dpi'] = fig.dpi
            
            plt.savefig(output_file, **save_kwargs)
            plt.close(fig)
        
        saved_files.append(output_file)
        print(f"Saved: {output_file}")
    
    return saved_files


def convert_to_math_text(eq_str):
    """
    Convert python equation string to matplotlib math text format
    
    Args:
        eq_str: str, equation string
        
    Returns:
        str: formatted equation string with matplotlib math text syntax
    """    
    # Remove Python-specific syntax
    eq_str = eq_str.replace('np.', '')
    eq_str = eq_str.replace('[i-1]', '_i')  # Directly to subscript
    
    # Convert to math notation
    eq_str = eq_str.replace(' * ', r' \cdot ')
    eq_str = eq_str.replace('arcsinh(', r'\mathrm{arcsinh}(')
    eq_str = eq_str.replace('arccosh(', r'\mathrm{arccosh}(')
    eq_str = eq_str.replace('arctanh(', r'\mathrm{arctanh}(')
    eq_str = eq_str.replace('arcsin(', r'\arcsin(')
    eq_str = eq_str.replace('arccos(', r'\arccos(')
    eq_str = eq_str.replace('arctan(', r'\arctan(')
    eq_str = eq_str.replace('sinh(', r'\sinh(')
    eq_str = eq_str.replace('cosh(', r'\cosh(')
    eq_str = eq_str.replace('tanh(', r'\tanh(')
    eq_str = eq_str.replace('sin(', r'\sin(')
    eq_str = eq_str.replace('cos(', r'\cos(')
    eq_str = eq_str.replace('tan(', r'\tan(')
    eq_str = eq_str.replace('exp(', r'\exp(')
    eq_str = eq_str.replace('pi', r'\pi')
    eq_str = re.sub(r'\*\*(\d+)', r'^{\1}', eq_str)
    
    return eq_str


def create_attractor_with_eq(data, output_file, equation_id,
                               point_size=POINT_SIZE_DEFAULT,
                               alpha=ALPHA_DEFAULT,
                               **plot_kwargs):
    """
    Create attractor visualization with equation info below
    
    Args:
        data: dict from prepare_attractor_data
        output_file: str, output filename
        equation_id: str, equation name
        point_size: float, marker size
        alpha: float, transparency
        **plot_kwargs: additional plotting parameters
        
    Returns:
        matplotlib figure
    """
    params = data['params']
    x = data['x']
    y = data['y']
    
    # Calculate ranges
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    aspect_ratio = x_range / y_range
    
    # Get equations for display
    x_eq_raw = EQUATION_LIBRARY[equation_id]['x_eq']
    y_eq_raw = EQUATION_LIBRARY[equation_id]['y_eq']

    # Convert to math text
    x_eq_math = convert_to_math_text(x_eq_raw)
    y_eq_math = convert_to_math_text(y_eq_raw)
    
    # Create figure with space for text below
    figsize = plot_kwargs.get('figsize', None)
    if figsize is None:
        width = 12
        height = width / aspect_ratio
        figsize = (width, height * 1.10)  # Add space for text
    
    dpi = plot_kwargs.get('dpi', 300)
    background_color = plot_kwargs.get('background_color', 'white')
    
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=background_color)
    
    # Create grid: 4 rows, main plot gets first 3, text gets last 1
    gs = fig.add_gridspec(2, 1, height_ratios=[9, 1], 
                          hspace=0.05, left=0.05, right=0.95, 
                          top=0.95, bottom=0.05)
    
    # Main attractor plot
    ax_main = fig.add_subplot(gs[0, 0])
    
    # Get colormap
    color_method = plot_kwargs.get('color_method', COLOR_METHOD)
    cmap = create_colormap(color_method, **plot_kwargs)
    
    # Plot attractor
    ax_main.scatter(x, y, 
                    c=data['density'], 
                    cmap=cmap,
                    s=point_size, 
                    alpha=alpha,
                    edgecolors='none',
                    rasterized=False)
    
    ax_main.set_aspect('equal')
    ax_main.axis('off')
    ax_main.set_facecolor(background_color)
    
    # Text panel
    ax_text = fig.add_subplot(gs[1, 0])
    ax_text.axis('off')
    ax_text.set_facecolor(background_color)
    
    info_lines = []
    info_lines.append(rf"$x_{{i+1}} = {x_eq_math}$")
    info_lines.append(rf"$y_{{i+1}} = {y_eq_math}$")
    info_lines.append("")  # blank line
    info_lines.append(f"$a = {params['a']},  b = {params['b']},  c = {params['c']},  d = {params['d']}$")
    info_lines.append(rf"$x_0 = {X_START},  y_0 = {Y_START}$")

    info_text = '\n'.join(info_lines)

    # Add text to panel - centered
    ax_text.text(0.5, 0.5, info_text,
            ha='center', va='center',
            fontsize=10,
            transform=ax_text.transAxes)
    
    # Save with format-specific settings
    fmt = os.path.splitext(output_file)[1][1:]  # Get extension without dot
    save_kwargs = {
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
        'facecolor': fig.get_facecolor()
    }
    
    if fmt == 'png':
        save_kwargs['dpi'] = dpi
    elif fmt == 'pdf':
        save_kwargs['dpi'] = dpi
    
    plt.savefig(output_file, **save_kwargs)
    plt.close(fig)
    
    return fig, output_file


# ===================================================
# Main Execution
# ===================================================

if __name__ == "__main__":
    # Set equation
    ACTIVE_EQUATION = "Tinkerbell"
    
    # Set parameters
    params = {
        'a': 0.9,
        'b': -0.6013,
        'c': 2,
        'd': 0.5
    }
    
    print(f"Generating attractor: {ACTIVE_EQUATION}")
    print(f"Parameters: a={params['a']}, b={params['b']}, c={params['c']}, d={params['d']}")
    print(f"Iterations: {ITERATIONS_DEFAULT:,}")
    
    # Generate data
    data = prepare_attractor_data(params, ACTIVE_EQUATION)
    
    print(f"Generated {len(data['x']):,} valid points")
    print(f"X range: [{data['x'].min():.3f}, {data['x'].max():.3f}]")
    print(f"Y range: [{data['y'].min():.3f}, {data['y'].max():.3f}]")
    
    # Save standard version (without equation)
    print("\nSaving standard version...")
    save_attractor(data, 
                  'attractor_output',
                  save_format='all',  # Will save PNG, PDF, and SVG
                  color_method=COLOR_METHOD,
                  palette=VIRIDIS_PALETTE,
                  low=GRADIENT_LOW,
                  high=GRADIENT_HIGH)
    
    # Save version with equation
    print("\nSaving version with equation...")
    save_attractor(data,
                  'attractor_with_eq',
                  save_format='all',
                  include_info=True,
                  equation_id=ACTIVE_EQUATION,
                  color_method=COLOR_METHOD,
                  palette=VIRIDIS_PALETTE,
                  low=GRADIENT_LOW,
                  high=GRADIENT_HIGH)
    
    print("\nDone!")
