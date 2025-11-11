"""
4 Parameter Chaotic Attractor Visualizer

This program generates and visualizes 4-parameter chaotic attractors.

Users can specify different chaotic attractor equations and parameters
to explore different attractor behaviors.

Author: Alex Spigler
"""

import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde

from .equations import EQUATION_LIBRARY

# ===================================================
# Configuration Parameters
# ===================================================

# Visualization settings
ALPHA_DEFAULT: float = 0.3  # 0.0 - 1.0 (fully opaque)
POINT_SIZE_DEFAULT: float = 0.15

# Color configuration
COLOR_METHOD: str = (
    "gradientn"  # Options: "viridis", "gradient", "gradient3", "gradientn"
)

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
    eq_str: str, param_names: Tuple[str, ...] = ("a", "b", "c", "d")
) -> Callable:
    """
    Compile equation string into a callable function.

    Returns a function that takes (x_prev, y_prev, a, b, c, d) and computes the next value.
    """
    # Replace array notation with previous-value names
    eq_str = eq_str.replace("x[n-1]", "x_prev")
    eq_str = eq_str.replace("y[n-1]", "y_prev")

    func_str = f"lambda x_prev, y_prev, a, b, c, d: {eq_str}"
    namespace = {"np": np, "__builtins__": {}}

    try:
        return eval(func_str, namespace)
    except Exception as e:
        raise ValueError(f"Failed to compile equation '{eq_str}': {e}") from e


def _get_equation_functions(equation_id: str) -> Tuple[Callable, Callable]:
    """Get compiled (x_update, y_update) functions for the specified equation."""
    if equation_id not in EQUATION_LIBRARY:
        raise KeyError(f"Unknown equation: '{equation_id}'. ")

    eq_dict = EQUATION_LIBRARY[equation_id]
    x_func = _compile_equation(eq_dict["x_eq"])
    y_func = _compile_equation(eq_dict["y_eq"])

    return x_func, y_func


# ===================================================
# Core Functions
# ===================================================


def generate_chaotic(
    params: Dict[str, float],
    equation_id: str,
    x_start: float,
    y_start: float,
    iterations: int = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Generate trajectory points for a chaotic attractor using iterative equations.
    """
    # Validate parameters
    required_params = {"a", "b", "c", "d"}
    if not required_params.issubset(params.keys()):
        missing = required_params - params.keys()
        raise KeyError(f"Missing required parameters: {missing}")

    a, b, c, d = params["a"], params["b"], params["c"], params["d"]

    # Pre-allocate arrays
    x = np.zeros(iterations, np.float64)
    y = np.zeros(iterations, np.float64)

    # Set initial conditions
    x[0] = x_start
    y[0] = y_start

    # Get compiled equation functions
    x_func, y_func = _get_equation_functions(equation_id)

    # Iterate through the dynamical system
    valid_length = iterations
    for n in range(1, iterations):
        try:
            x[n] = x_func(x[n - 1], y[n - 1], a, b, c, d)
            y[n] = y_func(x[n - 1], y[n - 1], a, b, c, d)

            # Early termination if values become non-finite
            if not (np.isfinite(x[n]) and np.isfinite(y[n])):
                valid_length = n
                break

        except (FloatingPointError, OverflowError):
            valid_length = n
            break

    x = x[:valid_length]
    y = y[:valid_length]

    return x, y


def evaluate_attractor_first(
    params: Dict[str, float],
    equation_id: str,
    x: np.ndarray = None,
    y: np.ndarray = None,
    x_start: float = None,
    y_start: float = None,
    min_small_side: float = 0.25,
    max_small_side: float = 500.0,
    max_aspect_ratio: float = 4.0,
    iterations: int = None,
) -> Dict[str, Any]:
    """
    Evaluate parameter set quality using geometric and statistical checks.

    Filters out uninteresting parameter sets. Checks for divergence, collapse to
    periodic orbits, and poor aspect ratios.

    Returns:
        Dictionary containing:
            - score: 2 if passed, -1 if rejected
            - reason: str explaining rejection or 'Passed all checks'
            - x_range: float (if passed)
            - y_range: float (if passed)
            - aspect_ratio: float (if passed)
    """
    # If data not provided, generate it
    if x is None or y is None:
        if params is None or equation_id is None:
            raise ValueError("Must provide either (x, y) or (params, equation_id)")

        try:
            x, y = generate_chaotic(
                params=params,
                equation_id=equation_id,
                iterations=iterations,
                x_start=x_start,
                y_start=y_start,
            )

        except Exception as e:
            return {
                "score": -1.0,
                "reason": f"Exception during generation: {e}",
            }

    x = np.asarray(x)
    y = np.asarray(y)

    # Remove non-finite values
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if x.size < 100 or y.size < 100:
        return {
            "score": -1.0,
            "reason": "Insufficient valid points",
        }

    x_diff = np.max(x) - np.min(x)
    y_diff = np.max(y) - np.min(y)

    # Range checks
    if x_diff < min_small_side or y_diff < min_small_side:
        return {
            "score": -1.0,
            "reason": f"Range too small ({'x_diff' if x_diff < y_diff else 'y_diff'} = {min(x_diff, y_diff):.2f}) - likely collapses",
        }

    if x_diff > max_small_side or y_diff > max_small_side:
        return {
            "score": -1.0,
            "reason": f"Range too large ({'x_diff' if x_diff > y_diff else 'y_diff'} = {max(x_diff, y_diff):.2f}), likely diverges",
        }

    aspect_ratio = x_diff / y_diff
    max_aspect_component = max(aspect_ratio, 1.0 / aspect_ratio)

    if max_aspect_component > max_aspect_ratio:
        return {
            "score": -1.0,
            "reason": f"Bad aspect ratio ({max_aspect_component:.2f})",
        }

    return {
        "score": 2,
        "reason": "Passed all checks",
        "x_range": x_diff,
        "y_range": y_diff,
        "aspect_ratio": aspect_ratio,
    }


def prepare_generate_data(
    params: Dict[str, float],
    x_start: float,
    y_start: float,
    equation_id: str,
    test_iterations: int = 100_000,
    final_iterations: int = 2_000_000,
    kde_sample_size: int = 50_000,
) -> Dict[str, Any]:
    """
    Generate attractor points and compute density using KDE for visualization.

    Returns dict with keys: 'x', 'y', 'density' (normalized 0-1), 'params'.
    Raises ValueError if fewer than 10,000 valid points are generated.
    """

    try:
        print("")
        print(
            f"Testing parameter set with {test_iterations:,} iterations before full generation..."
        )

        evaluation = evaluate_attractor_first(
            params=params,
            x_start=x_start,
            y_start=y_start,
            equation_id=equation_id,
            iterations=test_iterations,
        )

        # Skip if evaluation failed
        if evaluation["score"] < 0:
            raise ValueError(f"Attractor validation failed: {evaluation['reason']}")

        # Passed quick checks; now do full iterations and save
        print("Initial tests passed")
        print(f"Generating full attractor with {final_iterations:,} iterations")

        # Generate points
        x, y = generate_chaotic(
            params=params,
            equation_id=equation_id,
            iterations=final_iterations,
            x_start=x_start,
            y_start=y_start,
        )

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

        print("Computing kernel density estimation...")

        # Compute density using KDE
        try:
            # Sample for KDE efficiency
            sample_size = min(len(x), kde_sample_size)
            indices = np.random.choice(len(x), sample_size, replace=False)
            x_sample = x[indices]
            y_sample = y[indices]

            # Calculate KDE
            kde = gaussian_kde(np.vstack([x_sample, y_sample]))

            # Evaluate KDE on the sample
            density_sample = kde(np.vstack([x_sample, y_sample]))

            density = griddata(
                points=(x_sample, y_sample),
                values=density_sample,
                xi=(x, y),
                method="linear",
                fill_value=density_sample.min(),
            )

            print(f"Initial Density Range: {density.min():.4f} to {density.max():.4f}")

            # Normalize density to [0, 1]
            density_range = density.max() - density.min()
            if density_range != 0:
                density = (density - density.min()) / density_range
                print(
                    f"Normalized Density Range: {density.min():.2f} to {density.max():.2f}"
                )
            else:
                density = np.ones(len(x), np.float64)
                print("Warning: Density is a constant--normalized to 1")

        except Exception as e:
            print(f"Warning: Could not compute KDE density: {e}")
            print("Falling back to uniform density")
            density = np.ones(len(x), np.float64)

        return {
            "x": x,
            "y": y,
            "density": density,
            "params": params,
            "equation_id": equation_id,
        }

    except KeyboardInterrupt:
        print("\nInterrupted by user")


def create_colormap(method: str, **kwargs: Any) -> LinearSegmentedColormap:
    """
    Create colormap for attractor visualization.

    Methods: "viridis", "gradient", "gradient3", "gradientn".
    Pass method-specific colors/settings via kwargs.
    """
    if method == "viridis":
        palette = kwargs.get("palette", VIRIDIS_PALETTE)
        return colormaps[palette]

    elif method == "gradient":
        low = kwargs.get("low", GRADIENT_LOW)
        high = kwargs.get("high", GRADIENT_HIGH)
        return LinearSegmentedColormap.from_list("custom", [low, high])

    elif method == "gradient3":
        low = kwargs.get("low", GRADIENT3_LOW)
        mid = kwargs.get("mid", GRADIENT3_MID)
        high = kwargs.get("high", GRADIENT3_HIGH)
        midpoint = kwargs.get("midpoint", GRADIENT3_MIDPOINT)
        colors = [low, mid, high]
        positions = [0, midpoint, 1]
        return LinearSegmentedColormap.from_list("custom", list(zip(positions, colors)))

    elif method == "gradientn":
        colors = kwargs.get("colors", GRADIENTN_COLORS)
        values = kwargs.get("values", GRADIENTN_VALUES)
        return LinearSegmentedColormap.from_list("custom", list(zip(values, colors)))

    else:
        print(f"Warning: Unknown color method '{method}', defaulting to viridis")
        return colormaps["viridis"]


def plot_chaotic(
    data: Dict[str, Any],
    point_size: float = POINT_SIZE_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    background_color: str = "white",
    color_method: str = COLOR_METHOD,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    **color_kwargs: Any,
) -> Tuple[Figure, Axes]:
    """Create and return matplotlib figure and axes with the attractor plotted."""
    x = data["x"]
    y = data["y"]
    density = data["density"]

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
        x,
        y,
        c=density,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        edgecolors="none",
        rasterized=False,
    )

    # Formatting
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(background_color)

    return fig, ax


def save_attractor(
    data: Dict[str, Any],
    x_start: float,
    y_start: float,
    output_dir: str = "output",
    prefix: str | None = None,
    start_counter: int = 1,
    point_size: float = POINT_SIZE_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    save_format: str = "png",
    include_info: bool = True,
    **plot_kwargs: Any,
) -> List[str]:
    """
    Generate and save attractor visualization to file(s).

    Returns:
        List of saved file paths
    """
    try:

        # Get equation_id from data
        equation_id = data.get("equation_id")
        if include_info and equation_id is None:
            raise ValueError("include_info=True requires equation_id in data dict")

        # Set prefix to equation_id if not provided
        if prefix is None:
            if equation_id is None:
                raise ValueError(
                    "Either prefix must be provided or equation_id must be in data dict"
                )
            prefix = equation_id

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Construct base filename
        base_filename = f"{prefix}_{start_counter}"

        # Determine which formats to save
        if save_format == "all":
            formats = ["png", "pdf", "svg"]
        else:
            formats = [save_format]

        saved_files = []

        for fmt in formats:
            output_path = os.path.join(output_dir, f"{base_filename}.{fmt}")

            if include_info and equation_id:
                # Create figure with info panel
                fig = create_attractor_with_eq(
                    data=data,
                    equation_id=equation_id,
                    x_start=x_start,
                    y_start=y_start,
                    point_size=point_size,
                    alpha=alpha,
                    **plot_kwargs,
                )
            else:
                # Standard attractor without info
                fig, ax = plot_chaotic(
                    data=data, point_size=point_size, alpha=alpha, **plot_kwargs
                )

            # Format-specific settings
            save_kwargs = {
                "bbox_inches": "tight",
                "pad_inches": 0.05,
                "facecolor": fig.get_facecolor(),
            }

            if fmt in ("png", "pdf"):
                save_kwargs["dpi"] = fig.dpi

            plt.savefig(output_path, **save_kwargs)
            plt.close(fig)

            saved_files.append(output_path)
            print(f"Saved: {output_path}")

            print("\nDone!")

        return saved_files

    except KeyboardInterrupt:
        print("\nInterrupted by user")


def convert_to_math_text(eq_str: str) -> str:
    """Convert Python equation syntax to matplotlib math text for matplotlib rendering."""
    # Remove Python-specific syntax
    eq_str = eq_str.replace("np.", "")
    eq_str = eq_str.replace("[n-1]", "_n")  # Array index to subscript

    # Convert operators to LaTeX
    eq_str = eq_str.replace(" * ", "")
    eq_str = eq_str.replace("pi", r"\pi")

    # Convert functions to LaTeX commands (order matters!)
    function_map = {
        "arcsinh": r"\mathrm{arcsinh}",
        "arccosh": r"\mathrm{arccosh}",
        "arctanh": r"\mathrm{arctanh}",
        "arcsin": r"\arcsin",
        "arccos": r"\arccos",
        "arctan": r"\arctan",
        "sinh": r"\sinh",
        "cosh": r"\cosh",
        "tanh": r"\tanh",
        "sin": r"\sin",
        "cos": r"\cos",
        "tan": r"\tan",
        "exp": r"\exp",
    }

    for func, latex_func in function_map.items():
        eq_str = eq_str.replace(f"{func}(", f"{latex_func}(")

    # Convert exponentiation: **2 -> ^{2}
    eq_str = re.sub(r"\*\*(\d+)", r"^{\1}", eq_str)

    return eq_str


def create_attractor_with_eq(
    data: Dict[str, Any],
    equation_id: str,
    x_start: float,
    y_start: float,
    point_size: float = POINT_SIZE_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    **plot_kwargs: Any,
) -> Figure:
    """
    Create attractor visualization with equation and parameter info panel below.

    Returns matplotlib Figure (does not save to file).
    """
    params = data["params"]
    x = data["x"]
    y = data["y"]

    # Calculate aspect ratio
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    if y_range == 0 or x_range == 0:
        aspect_ratio = 1.0
    else:
        aspect_ratio = x_range / y_range

    # Get equations for display
    x_eq_raw = EQUATION_LIBRARY[equation_id]["x_eq"]
    y_eq_raw = EQUATION_LIBRARY[equation_id]["y_eq"]
    # Convert to LaTeX math text
    x_eq_math = convert_to_math_text(x_eq_raw)
    y_eq_math = convert_to_math_text(y_eq_raw)

    # Create figure with space for text panel below
    figsize = plot_kwargs.get("figsize", None)
    if figsize is None:
        width = 12
        height = width / aspect_ratio
        figsize = (width, height * 1.10)  # Add 10% space for text

    dpi = plot_kwargs.get("dpi", 300)
    background_color = plot_kwargs.get("background_color", "white")

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=background_color)

    # Create grid: main plot gets 90%, text panel gets 10%
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[9, 1],
        hspace=0.05,
        left=0.05,
        right=0.95,
        top=0.95,
        bottom=0.05,
    )

    # Main attractor plot
    ax_main = fig.add_subplot(gs[0, 0])

    # Get colormap
    color_method = plot_kwargs.get("color_method", COLOR_METHOD)
    cmap = create_colormap(color_method, **plot_kwargs)

    # Plot attractor
    ax_main.scatter(
        x,
        y,
        c=data["density"],
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        edgecolors="none",
        rasterized=False,
    )

    ax_main.set_aspect("equal")
    ax_main.axis("off")
    ax_main.set_facecolor(background_color)

    # Text panel for equations and parameters
    ax_text = fig.add_subplot(gs[1, 0])
    ax_text.axis("off")
    ax_text.set_facecolor(background_color)

    # Build info text
    info_lines = [
        rf"$x_{{n+1}} = {x_eq_math}$",
        rf"$y_{{n+1}} = {y_eq_math}$",
        "",  # Blank line for spacing
        f"$a = {params['a']},  b = {params['b']},  c = {params['c']},  d = {params['d']}$",
        rf"$x_0 = {x_start},  y_0 = {y_start}$",
    ]
    info_text = "\n".join(info_lines)

    # Add centered text to panel
    ax_text.text(
        0.5,
        0.5,
        info_text,
        ha="center",
        va="center",
        fontsize=10,
        transform=ax_text.transAxes,
    )

    return fig
