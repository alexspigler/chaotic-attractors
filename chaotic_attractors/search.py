"""
Parameter search for 4-parameter chaotic attractors.

This module provides:
- generate_random: uniformly sample parameters with fixed decimal places
- evaluate_attractor_second: quick screening of full parameter sets
- search_attractors: random search over parameter space
- prepare_search_data: scoring results
"""

import csv
import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde

from .core import (
    COLOR_METHOD,
    GRADIENT_HIGH,
    GRADIENT_LOW,
    VIRIDIS_PALETTE,
    evaluate_attractor_first,
    generate_chaotic,
    save_attractor,
)


def generate_random(min_val: float, max_val: float, decimals: int) -> float:
    """
    Generate a random value with fixed decimal precision.

    Samples uniformly from the discrete set of values between min_val and max_val
    that can be represented with the specified number of decimal places.


    Returns:
        Random value rounded to specified decimal places
    """
    step = 10 ** (-decimals)
    num_steps = int(round((max_val - min_val) / step))

    if num_steps <= 0:
        raise ValueError(
            f"Invalid range: min_val={min_val}, max_val={max_val}, step={step}"
        )

    index = np.random.randint(0, num_steps + 1)
    value = min_val + index * step
    return round(value, decimals)


def search_attractors(
    equation_id: str,
    x_start: float,
    y_start: float,
    num_to_find: int = 10,
    max_attempts: int = 5_000,
    parameter_ranges: Dict[str, Tuple[float, float]] = None,
    test_iterations: int = 100_000,
    final_iterations: int = 2_000_000,
    decimals: int = 2,
    output_dir: str = "output",
    prefix: str | None = None,
    progress_interval: int = 50,
    start_counter: int = 1,
    min_small_side: float = 0.25,
    max_small_side: float = 500.0,
    digits_unique: int = 4,
    min_unique_ratio: float = 0.25,
    max_unique_ratio: float = 1.0,
    max_aspect_ratio: float = 4.0,
    include_info: bool = True,
    save_format: str = "png",
) -> Dict[str, Any]:
    """
    Perform random search over parameter space to find interesting attractors.

    Randomly samples parameter combinations, evaluates them with quick tests,
    then generates and saves full-resolution visualizations of those that pass.
    Creates both individual files and a summary CSV.

    Returns:
        Dictionary containing:
            - attractors: List[Dict] with parameters, stats, and filenames
            - summary: List[Dict] with rows for CSV output
            - attempts: int total parameter sets tested
            - elapsed_minutes: float total search time
    """
    if parameter_ranges is None:
        parameter_ranges = {
            "a": (-3, 3),
            "b": (-3, 3),
            "c": (-3, 3),
            "d": (-3, 3),
        }

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set prefix to equation_id if not provided
    if prefix is None:
        prefix = equation_id

    found_attractors = []
    attempts = 0

    print("")
    print(f"Starting search for {num_to_find} attractors (equation: {equation_id})...")
    start_time = time.time()

    try:
        while len(found_attractors) < num_to_find and attempts < max_attempts:
            attempts += 1

            # Sample parameters with limited decimals
            params = {
                "a": generate_random(
                    parameter_ranges["a"][0], parameter_ranges["a"][1], decimals
                ),
                "b": generate_random(
                    parameter_ranges["b"][0], parameter_ranges["b"][1], decimals
                ),
                "c": generate_random(
                    parameter_ranges["c"][0], parameter_ranges["c"][1], decimals
                ),
                "d": generate_random(
                    parameter_ranges["d"][0], parameter_ranges["d"][1], decimals
                ),
            }

            attractor_num = len(found_attractors) + start_counter

            evaluation = evaluate_attractor_first(
                params=params,
                x_start=x_start,
                y_start=y_start,
                equation_id=equation_id,
                iterations=test_iterations,
            )

            if attempts % progress_interval == 0:
                elapsed_sec = time.time() - start_time
                rate = elapsed_sec / attempts if attempts > 0 else 0.0
                print(
                    f"\rAttempts: {attempts} | "
                    f"Found: {len(found_attractors)}/{num_to_find} | "
                    f"Elapsed: {elapsed_sec:.1f} sec | Rate: {1/rate:.2f} test/sec",
                    flush=True,
                )

            # Skip if evaluation failed
            if evaluation["score"] < 0:
                continue

            # Passed quick checks
            print("\n\nInitial tests passed")
            print(
                f"Found candidate {attractor_num} at attempt {attempts} "
                f"with params: a={params['a']}, b={params['b']}, "
                f"c={params['c']}, d={params['d']}"
            )
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

            # second evaluation
            evaluation = evaluate_attractor_second(
                params=params,
                equation_id=equation_id,
                x=x,
                y=y,
                min_small_side=min_small_side,
                max_small_side=max_small_side,
                digits_unique=digits_unique,
                min_unique_ratio=min_unique_ratio,
                max_unique_ratio=max_unique_ratio,
                max_aspect_ratio=max_aspect_ratio,
            )

            # Skip if evaluation failed
            if evaluation["score"] < 0:
                print(f"Attractor validation failed: {evaluation['reason']}")
                print("")
                continue

            try:
                data = prepare_search_data(
                    params=params,
                    x=x,
                    y=y,
                    final_iterations=final_iterations,
                    kde_sample_size=50_000,
                    equation_id=equation_id,
                )

            except Exception as e:
                print(f"Skipped attractor {attractor_num} due to data prep error: {e}")
                continue

            # Passed final checks
            print("Final tests passed")
            print(
                f"Score: {evaluation['score']:.2f} | "
                f"X range: {evaluation['x_range']:.3f} | "
                f"Y range: {evaluation['y_range']:.3f} | "
                f"Aspect ratio: {evaluation['aspect_ratio']:.2f} | "
                f"Unique ratio: {evaluation['unique_ratio']:.3f}"
            )

            # Save attractor image
            saved_files = save_attractor(
                data=data,
                x_start=x_start,
                y_start=y_start,
                output_dir=output_dir,
                prefix=prefix,
                start_counter=attractor_num,
                save_format=save_format,
                include_info=include_info,
                color_method=COLOR_METHOD,
                palette=VIRIDIS_PALETTE,
                low=GRADIENT_LOW,
                high=GRADIENT_HIGH,
            )

            found_attractors.append(
                {
                    "parameters": params,
                    "evaluation": evaluation,
                    "files": saved_files,
                }
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        elapsed_minutes = (time.time() - start_time) / 60.0
        print(
            f"\n\n{len(found_attractors)} attractors completed after {attempts} attempts "
            f"({elapsed_minutes:.2f} minutes) "
            f"Rate: {elapsed_minutes/num_to_find:.2f} minutes/test"
        )

        # Build summary table
        summary_rows = []
        for index, attr in enumerate(found_attractors, start=start_counter):
            eval_ = attr["evaluation"]
            row = {
                "name": f"{prefix}_{index}",
                "a": attr["parameters"]["a"],
                "b": attr["parameters"]["b"],
                "c": attr["parameters"]["c"],
                "d": attr["parameters"]["d"],
                "score": eval_["score"],
                "x_range": eval_["x_range"],
                "y_range": eval_["y_range"],
                "aspect_ratio": eval_["aspect_ratio"],
                "unique_ratio": eval_["unique_ratio"],
            }
            summary_rows.append(row)

        # Write CSV summary
        if summary_rows:
            csv_path = os.path.join(output_dir, f"{prefix}_summary.csv")
            fieldnames = [
                "name",
                "a",
                "b",
                "c",
                "d",
                "score",
                "x_range",
                "y_range",
                "aspect_ratio",
                "unique_ratio",
            ]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in summary_rows:
                    writer.writerow(row)
            print(f"Saved parameter summary to: {csv_path}")
        else:
            print("No attractors found that met the criteria.")

        result = {
            "attractors": found_attractors,
            "summary": summary_rows,
            "attempts": attempts,
            "elapsed_minutes": elapsed_minutes,
        }

    return result


def prepare_search_data(
    params: Dict[str, float],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    equation_id: str,
    final_iterations: int = 2_000_000,
    kde_sample_size: int = 50_000,
) -> Dict[str, Any]:
    """
    Compute density using KDE for visualization.

    Returns dict with keys: 'x', 'y', 'density' (normalized 0-1), 'params'.
    Raises ValueError if fewer than 10,000 valid points are generated.
    """
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


def evaluate_attractor_second(
    params: Dict[str, float],
    equation_id: str,
    x: np.ndarray = None,
    y: np.ndarray = None,
    x_start: float = None,
    y_start: float = None,
    min_small_side: float = 0.25,
    max_small_side: float = 500.0,
    digits_unique: int = 4,
    min_unique_ratio: float = 0.25,
    max_unique_ratio: float = 1.0,
    max_aspect_ratio: float = 4.0,
    iterations: int = None,
) -> Dict[str, Any]:
    """
    Evaluate parameter set quality using geometric and statistical checks.

    Filters out uninteresting parameter sets. Checks for divergence, collapse to
    periodic orbits, and poor aspect ratios.

    Returns:
        Dictionary containing:
            - score: float >= 0 if passed (lower is better), -1 if rejected
            - reason: str explaining rejection or 'Passed all checks'
            - x_range: float (if passed)
            - y_range: float (if passed)
            - unique_ratio: float (if passed)
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

    # Unique ratio
    rounded_points = np.column_stack(
        [np.round(x, digits_unique), np.round(y, digits_unique)]
    )
    unique_points = np.unique(rounded_points, axis=0)
    unique_ratio = unique_points.shape[0] / rounded_points.shape[0]

    if unique_ratio < min_unique_ratio:
        return {
            "score": -1.0,
            "reason": f"Low unique point ratio ({unique_ratio:.4f}) - likely collapses",
        }

    if unique_ratio > max_unique_ratio:
        return {
            "score": -1.0,
            "reason": f"High unique point ratio ({unique_ratio:.4f}) - likely diverges",
        }

    aspect_ratio = x_diff / y_diff
    max_aspect_component = max(aspect_ratio, 1.0 / aspect_ratio)

    if max_aspect_component > max_aspect_ratio:
        return {
            "score": -1.0,
            "reason": f"Bad aspect ratio ({max_aspect_component:.2f})",
        }

    # Scoring (0 is best)
    # Measures normalized squared deviations from ideal characteristics
    ideal_aspect_ratio = 3 / 2  # Prefer slightly rectangular attractors
    ideal_unique_ratio = 2 / 3  # Balance between structure and complexity
    min_aspect_ratio = 1.0

    # Calculate maximum possible deviation in either direction from ideal
    # This ensures symmetric penalization regardless of which side of ideal
    max_aspect_deviation = max(
        abs(min_aspect_ratio - ideal_aspect_ratio),
        abs(max_aspect_ratio - ideal_aspect_ratio),
    )

    max_unique_deviation = max(
        abs(min_unique_ratio - ideal_unique_ratio),
        abs(max_unique_ratio - ideal_unique_ratio),
    )

    # Normalize deviations to [0, 1] scale (after squaring)
    aspect_normalized = (
        max_aspect_component - ideal_aspect_ratio
    ) / max_aspect_deviation
    unique_normalized = (unique_ratio - ideal_unique_ratio) / max_unique_deviation

    # Range: [0, 2] where 0 = both metrics at ideal, 2 = both at worst extremes (that would pass filtering)
    score_final = aspect_normalized**2 + unique_normalized**2

    return {
        "score": score_final,
        "reason": "Passed all checks",
        "x_range": x_diff,
        "y_range": y_diff,
        "unique_ratio": unique_ratio,
        "aspect_ratio": aspect_ratio,
    }
