"""
4 Parameter Chaotic Attractor Visualizer

This program generates and visualizes 4-parameter chaotic attractors.

Users can specify different chaotic attractor equations and parameters
to explore different attractor behaviors.

Author: Alex Spigler
"""

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from scipy.interpolate import griddata
from scipy.spatial import QhullError
from scipy.stats import gaussian_kde

from .equations import EQUATION_LIBRARY

# ===================================================
# Configuration Parameters
# ===================================================

# Visualization settings
ALPHA_DEFAULT: float = 0.3  # opacity: 0.0 (transparent) to 1.0 (opaque)
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
GRADIENTN_COLORS: list[str] = ["aliceblue", "lightblue", "darkviolet", "purple"]
GRADIENTN_VALUES: list[float] = [0, 0.2, 0.65, 1]


# ===================================================
# Equation Compilation
# ===================================================


def _compile_equation(eq_str: str) -> Callable[..., float]:
    """
    Compile equation string into a callable function.

    Returns a function that takes (x_prev, y_prev, a, b, c, d) and computes the next value.
    """
    # Replace array notation with previous-value names
    eq_str = eq_str.replace("x[n-1]", "x_prev")
    eq_str = eq_str.replace("y[n-1]", "y_prev")

    func_str = f"lambda x_prev, y_prev, a, b, c, d: {eq_str}"
    # Safe to eval: eq_str comes only from the hardcoded EQUATION_LIBRARY (never
    # user input), and the namespace exposes only NumPy with built-ins disabled.
    namespace = {"np": np, "__builtins__": {}}

    try:
        return eval(func_str, namespace)
    except Exception as e:
        raise ValueError(f"Failed to compile equation '{eq_str}': {e}") from e


def _get_equation_functions(
    equation_id: str,
) -> tuple[Callable[..., float], Callable[..., float]]:
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
    params: Mapping[str, float],
    equation_id: str,
    x_start: float,
    y_start: float,
    iterations: int | None = None,
    burn_in: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Generate trajectory points for a chaotic attractor using iterative equations.
    """
    required_params = {"a", "b", "c", "d"}
    if not required_params.issubset(params.keys()):
        missing = required_params - params.keys()
        raise KeyError(f"Missing required parameters: {missing}")

    a, b, c, d = params["a"], params["b"], params["c"], params["d"]

    if iterations is None:
        raise ValueError("generate_chaotic() requires an explicit iterations count")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if burn_in < 0:
        raise ValueError("burn_in cannot be negative")
    if not np.isfinite(x_start) or not np.isfinite(y_start):
        raise ValueError("starting values must be finite")
    if not all(np.isfinite(value) for value in (a, b, c, d)):
        raise ValueError("parameters must be finite")

    total_iterations = iterations + burn_in
    x = np.zeros(total_iterations, np.float64)
    y = np.zeros(total_iterations, np.float64)

    x[0] = x_start
    y[0] = y_start

    x_func, y_func = _get_equation_functions(equation_id)

    valid_length = total_iterations
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for n in range(1, total_iterations):
            try:
                x[n] = x_func(x[n - 1], y[n - 1], a, b, c, d)
                y[n] = y_func(x[n - 1], y[n - 1], a, b, c, d)
            except (FloatingPointError, OverflowError):
                valid_length = n
                break

            if not (np.isfinite(x[n]) and np.isfinite(y[n])):
                valid_length = n
                break

    return x[burn_in:valid_length], y[burn_in:valid_length]


def _screen_geometry(
    params: Mapping[str, float],
    equation_id: str,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    x_start: float | None = None,
    y_start: float | None = None,
    min_small_side: float = 0.25,
    max_small_side: float = 500.0,
    iterations: int | None = None,
    burn_in: int = 0,
) -> dict[str, Any]:
    """
    Shared first-stage screening for both evaluators: generate the trajectory if
    (x, y) are not supplied, drop non-finite points, and apply the point-count and
    range checks.

    Returns either a rejection dict {"score": -1.0, "reason": ...} or a pass dict
    {"x", "y", "x_diff", "y_diff"}. The aspect-ratio and unique-ratio checks are
    left to the callers, which apply them in different orders.
    """
    if min_small_side <= 0 or max_small_side <= min_small_side:
        raise ValueError(
            "range bounds must satisfy 0 < min_small_side < max_small_side"
        )
    if (x is None) != (y is None):
        raise ValueError("x and y must be supplied together")

    if x is None or y is None:
        if x_start is None or y_start is None:
            raise ValueError("x_start and y_start are required when generating data")

        x, y = generate_chaotic(
            params=params,
            equation_id=equation_id,
            iterations=iterations,
            x_start=x_start,
            y_start=y_start,
            burn_in=burn_in,
        )

        if iterations is not None and len(x) != iterations:
            return {
                "score": -1.0,
                "reason": (
                    f"Trajectory terminated early ({len(x):,} of "
                    f"{iterations:,} points)"
                ),
            }

    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be one-dimensional arrays of equal length")

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

    return {
        "x": x,
        "y": y,
        "x_diff": x_diff,
        "y_diff": y_diff,
    }


def evaluate_attractor_first(
    params: Mapping[str, float],
    equation_id: str,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    x_start: float | None = None,
    y_start: float | None = None,
    min_small_side: float = 0.25,
    max_small_side: float = 500.0,
    max_aspect_ratio: float = 4.0,
    iterations: int | None = None,
    burn_in: int = 0,
) -> dict[str, Any]:
    """
    Evaluate parameter set quality using geometric and statistical checks.

    Filters out uninteresting parameter sets. Checks for divergence, collapse to
    periodic orbits, and poor aspect ratios.

    Returns:
        Dictionary containing:
            - score: 2.0 if passed, -1.0 if rejected
            - reason: str explaining rejection or 'Passed all checks'
            - x_range: float (if passed)
            - y_range: float (if passed)
            - aspect_ratio: float (if passed)
    """
    screen = _screen_geometry(
        params=params,
        equation_id=equation_id,
        x=x,
        y=y,
        x_start=x_start,
        y_start=y_start,
        min_small_side=min_small_side,
        max_small_side=max_small_side,
        iterations=iterations,
        burn_in=burn_in,
    )
    if "score" in screen:
        return screen

    x_diff = screen["x_diff"]
    y_diff = screen["y_diff"]

    aspect_ratio = max(x_diff, y_diff) / min(x_diff, y_diff)

    if aspect_ratio > max_aspect_ratio:
        return {
            "score": -1.0,
            "reason": f"Bad aspect ratio ({aspect_ratio:.2f})",
        }

    return {
        "score": 2.0,
        "reason": "Passed all checks",
        "x_range": x_diff,
        "y_range": y_diff,
        "aspect_ratio": aspect_ratio,
    }


def _compute_density(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    kde_sample_size: int = 50_000,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """
    Estimate per-point density with a Gaussian KDE.

    The KDE is fit and evaluated on a random subsample, then interpolated
    across the full trajectory. Subsampling keeps the kernel evaluation at
    O(sample^2) instead of O(n^2) on the full (up to multi-million) point set.
    Returns density normalized to [0, 1], or uniform density if the estimate
    cannot be computed.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError(
            "x and y must be non-empty one-dimensional arrays of equal length"
        )
    if kde_sample_size < 2:
        raise ValueError("kde_sample_size must be at least 2")

    rng = np.random.default_rng() if rng is None else rng
    print("Computing kernel density estimation...")
    try:
        sample_size = min(len(x), kde_sample_size)
        indices = rng.choice(len(x), sample_size, replace=False)
        x_sample = x[indices]
        y_sample = y[indices]

        kde = gaussian_kde(np.vstack([x_sample, y_sample]))
        density_sample = kde(np.vstack([x_sample, y_sample]))

        density = griddata(
            points=(x_sample, y_sample),
            values=density_sample,
            xi=(x, y),
            method="linear",
            fill_value=density_sample.min(),
        )
        print(f"Initial Density Range: {density.min():.4f} to {density.max():.4f}")

        density_range = density.max() - density.min()
        if density_range != 0:
            density = (density - density.min()) / density_range
            print(
                f"Normalized Density Range: {density.min():.2f} to {density.max():.2f}"
            )
        else:
            density = np.ones(len(x), np.float64)
            print("Warning: Density is a constant--normalized to 1")

    except (np.linalg.LinAlgError, QhullError, ValueError) as error:
        print(f"Warning: Could not compute KDE density: {error}")
        print("Falling back to uniform density")
        density = np.ones(len(x), np.float64)

    return density


def prepare_generate_data(
    params: Mapping[str, float],
    x_start: float,
    y_start: float,
    equation_id: str,
    test_iterations: int = 100_000,
    final_iterations: int = 2_000_000,
    kde_sample_size: int = 50_000,
    burn_in: int = 1_000,
    lyapunov_iterations: int = 10_000,
    min_lyapunov_exponent: float = 0.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Generate attractor points and compute density using KDE for visualization.

    Returns dict with keys: 'x', 'y', 'density' (normalized 0-1), 'params'.
    Raises ValueError if fewer than 10,000 valid points are generated.
    """

    if test_iterations <= 0 or final_iterations <= 0:
        raise ValueError("test_iterations and final_iterations must be positive")
    if final_iterations < 10_000:
        raise ValueError("final_iterations must be at least 10,000")

    seed_sequence = np.random.SeedSequence(seed)
    effective_seed = int(seed_sequence.entropy)
    rng = np.random.default_rng(seed_sequence)

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
        burn_in=burn_in,
    )

    if evaluation["score"] < 0:
        raise ValueError(f"Attractor validation failed: {evaluation['reason']}")

    # Passed quick checks; now do full iterations and save
    print("Initial tests passed")
    print(f"Generating full attractor with {final_iterations:,} iterations")

    x, y = generate_chaotic(
        params=params,
        equation_id=equation_id,
        iterations=final_iterations,
        x_start=x_start,
        y_start=y_start,
        burn_in=burn_in,
    )

    print(f"Generated {len(x):,} valid points")
    print(f"X range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Y range: [{y.min():.3f}, {y.max():.3f}]")

    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]

    if len(x) != final_iterations:
        raise ValueError(
            "Full trajectory terminated early: "
            f"generated {len(x):,} of {final_iterations:,} requested points"
        )

    from .metrics import estimate_largest_lyapunov

    lyapunov_exponent = estimate_largest_lyapunov(
        x=x,
        y=y,
        params=params,
        equation_id=equation_id,
        max_iterations=lyapunov_iterations,
    )
    if not np.isfinite(lyapunov_exponent) or lyapunov_exponent <= min_lyapunov_exponent:
        raise ValueError(
            "Attractor validation failed: finite-time Lyapunov exponent "
            f"{lyapunov_exponent:.4f} is not above {min_lyapunov_exponent:.4f}"
        )

    density = _compute_density(x, y, kde_sample_size, rng=rng)

    return {
        "x": x,
        "y": y,
        "density": density,
        "params": params,
        "equation_id": equation_id,
        "burn_in": burn_in,
        "lyapunov_exponent": lyapunov_exponent,
        "seed": effective_seed,
    }


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
        return LinearSegmentedColormap.from_list(
            "custom", list(zip(positions, colors, strict=True))
        )

    elif method == "gradientn":
        colors = kwargs.get("colors", GRADIENTN_COLORS)
        values = kwargs.get("values", GRADIENTN_VALUES)
        return LinearSegmentedColormap.from_list(
            "custom", list(zip(values, colors, strict=True))
        )

    else:
        print(f"Warning: Unknown color method '{method}', defaulting to viridis")
        return colormaps["viridis"]


def plot_chaotic(
    data: Mapping[str, Any],
    point_size: float = POINT_SIZE_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    background_color: str = "white",
    color_method: str = COLOR_METHOD,
    figsize: tuple[float, float] | None = None,
    dpi: int = 300,
    rasterized_points: bool = True,
    **color_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create and return matplotlib figure and axes with the attractor plotted."""
    x = data["x"]
    y = data["y"]
    density = data["density"]

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    if y_range == 0 or x_range == 0:
        aspect_ratio = 1.0
    else:
        aspect_ratio = x_range / y_range

    if figsize is None:
        width = 12
        height = width / aspect_ratio
        figsize = (width, height)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=background_color)

    cmap = create_colormap(color_method, **color_kwargs)

    ax.scatter(
        x,
        y,
        c=density,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        edgecolors="none",
        rasterized=rasterized_points,
    )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(background_color)

    return fig, ax


def save_attractor(
    data: Mapping[str, Any],
    x_start: float,
    y_start: float,
    output_dir: str = "output",
    prefix: str | None = None,
    start_counter: int = 1,
    point_size: float = POINT_SIZE_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    save_format: str = "png",
    include_info: bool = True,
    overwrite: bool = False,
    **plot_kwargs: Any,
) -> list[str]:
    """
    Generate and save attractor visualization to file(s).

    Returns:
        List of saved file paths
    """
    try:
        equation_id = data.get("equation_id")
        if include_info and equation_id is None:
            raise ValueError("include_info=True requires equation_id in data dict")

        if prefix is None:
            if equation_id is None:
                raise ValueError(
                    "Either prefix must be provided or equation_id must be in data dict"
                )
            prefix = equation_id

        output_directory = Path(output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)

        base_filename = f"{prefix}_{start_counter}"

        if save_format == "all":
            formats = ["png", "pdf", "svg"]
        else:
            formats = [save_format]

        supported_formats = {"png", "pdf", "svg"}
        if not set(formats).issubset(supported_formats):
            raise ValueError("save_format must be 'png', 'pdf', 'svg', or 'all'")

        output_paths = [output_directory / f"{base_filename}.{fmt}" for fmt in formats]
        existing = [path for path in output_paths if path.exists()]
        if existing and not overwrite:
            names = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Refusing to overwrite existing output: {names}. "
                "Pass overwrite=True to replace it."
            )

        saved_files = []

        for fmt, output_path in zip(formats, output_paths, strict=True):

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
                fig, _ = plot_chaotic(
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

            fig.savefig(output_path, **save_kwargs)
            plt.close(fig)

            saved_files.append(str(output_path))
            print(f"Saved: {output_path}")

        print("\nDone!")

        return saved_files

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        raise


def convert_to_math_text(eq_str: str) -> str:
    """Convert Python equation syntax to matplotlib math text for matplotlib rendering."""
    eq_str = eq_str.replace("np.", "")
    eq_str = eq_str.replace("[n-1]", "_n")  # Array index to subscript

    eq_str = eq_str.replace(" * ", "")
    eq_str = eq_str.replace("pi", r"\pi")

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

    function_pattern = re.compile(
        r"\b(" + "|".join(sorted(function_map, key=len, reverse=True)) + r")\("
    )
    eq_str = function_pattern.sub(
        lambda match: f"{function_map[match.group(1)]}(", eq_str
    )

    # Convert exponentiation: **2 -> ^{2}
    eq_str = re.sub(r"\*\*(\d+)", r"^{\1}", eq_str)

    return eq_str


def create_attractor_with_eq(
    data: Mapping[str, Any],
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

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    if y_range == 0 or x_range == 0:
        aspect_ratio = 1.0
    else:
        aspect_ratio = x_range / y_range

    x_eq_raw = EQUATION_LIBRARY[equation_id]["x_eq"]
    y_eq_raw = EQUATION_LIBRARY[equation_id]["y_eq"]
    x_eq_math = convert_to_math_text(x_eq_raw)
    y_eq_math = convert_to_math_text(y_eq_raw)

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

    ax_main = fig.add_subplot(gs[0, 0])

    color_method = plot_kwargs.get("color_method", COLOR_METHOD)
    cmap = create_colormap(color_method, **plot_kwargs)

    ax_main.scatter(
        x,
        y,
        c=data["density"],
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        edgecolors="none",
        rasterized=plot_kwargs.get("rasterized_points", True),
    )

    ax_main.set_aspect("equal")
    ax_main.axis("off")
    ax_main.set_facecolor(background_color)

    ax_text = fig.add_subplot(gs[1, 0])
    ax_text.axis("off")
    ax_text.set_facecolor(background_color)

    info_lines = [
        rf"$x_{{n+1}} = {x_eq_math}$",
        rf"$y_{{n+1}} = {y_eq_math}$",
        "",  # Blank line for spacing
        f"$a = {params['a']},  b = {params['b']},  c = {params['c']},  d = {params['d']}$",
        rf"$x_0 = {x_start},  y_0 = {y_start}$",
    ]
    info_text = "\n".join(info_lines)

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
