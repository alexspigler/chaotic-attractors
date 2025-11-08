"""
Parameter search for 4-parameter chaotic attractors.

This module provides:
- generate_random: sample parameters with fixed decimal places
- evaluate_attractor: quick screening of parameter sets
- search_attractors: random search over parameter space with filtering and saving
"""

import os
import time
import csv
from typing import Dict, Tuple, List, Any

import numpy as np

from .core import (
    X_START,
    Y_START,
    ITERATIONS_DEFAULT,
    COLOR_METHOD,
    VIRIDIS_PALETTE,
    GRADIENT_LOW,
    GRADIENT_HIGH,
    generate_chaotic,
    prepare_attractor_data,
    save_attractor,
)
from .equations import EQUATION_LIBRARY


def generate_random(min_val: float, max_val: float, decimals: int) -> float:
    """
    Generate a random value with fixed decimal precision.
    
    Samples uniformly from the discrete set of values between min_val and max_val
    that can be represented with the specified number of decimal places.
    
    Args:
        min_val: Lower bound of the range (inclusive)
        max_val: Upper bound of the range (inclusive)
        decimals: Number of decimal places to round to
        
    Returns:
        Random value rounded to specified decimal places
        
    Raises:
        ValueError: If range is invalid given the step size
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


def evaluate_attractor(
    params: Dict[str, float],
    equation_id: str,
    x_start: float = X_START,
    y_start: float = Y_START,
    test_iterations: int = 25_000,
    min_small_side: float = 0.25,
    max_small_side: float = 500.0,
    digits_unique: int = 2,
    min_unique_ratio: float = 0.1,
    max_unique_ratio: float = 0.95,
    max_aspect_ratio: float = 4.0,
) -> Dict[str, Any]:
    """
    Evaluate parameter set quality using geometric and statistical checks.
    
    Performs a quick evaluation with reduced iterations to filter out uninteresting
    parameter sets before full visualization. Checks for divergence, collapse to
    periodic orbits, and poor aspect ratios.
    
    Args:
        params: Dictionary with keys 'a', 'b', 'c', 'd'
        equation_id: Name of equation system from EQUATION_LIBRARY
        test_iterations: Number of points to generate for evaluation
        min_small_side: Minimum acceptable range in either dimension
        max_small_side: Maximum acceptable range (rejects divergent systems)
        digits_unique: Decimal places for uniqueness check
        min_unique_ratio: Minimum fraction of unique points (rejects periodic orbits)
        max_unique_ratio: Maximum fraction of unique points (rejects noise)
        max_aspect_ratio: Maximum ratio of longer side to shorter side
        
    Returns:
        Dictionary containing:
            - score: float >= 0 if passed (lower is better), -1 if rejected
            - reason: str explaining rejection or 'Passed all checks'
            - x_range: float (if passed)
            - y_range: float (if passed)
            - unique_ratio: float (if passed)
            - aspect_ratio: float (if passed)
    """
    try:
        x, y = generate_chaotic(params, equation_id, n_iterations=test_iterations,
                                x_start=x_start, y_start=y_start)
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
            "reason": "Too few points generated",
        }

    x_diff = np.max(x) - np.min(x)
    y_diff = np.max(y) - np.min(y)

    # Range checks
    if x_diff < min_small_side or y_diff < min_small_side:
        return {
            "score": -1.0,
            "reason": "Range too small - likely collapsed to a point or line",
        }
    
    if x_diff > max_small_side or y_diff > max_small_side:
        return {
            "score": -1.0,
            "reason": "Range too large, likely diverges",
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
            "reason": "Low unique point ratio - likely a periodic orbit",
        }

    if unique_ratio > max_unique_ratio:
        return {
            "score": -1.0,
            "reason": "High unique point ratio - too spread out / not interesting",
        }

    aspect_ratio = x_diff / y_diff
    max_aspect_component = max(aspect_ratio, 1.0 / aspect_ratio)

    if max_aspect_component > max_aspect_ratio:
        return {
            "score": -1.0,
            "reason": "Bad aspect ratio",
        }

    # Compute score using MSE from ideal values (lower is better)
    ideal_aspect_ratio = 1.5
    ideal_unique_ratio = 0.75

    aspect_normalized = (max_aspect_component - ideal_aspect_ratio) / (max_aspect_ratio - 1.0)
    unique_normalized = (unique_ratio - ideal_unique_ratio) / (max_unique_ratio - min_unique_ratio)

    score = aspect_normalized**2 + unique_normalized**2

    return {
        "score": score,
        "reason": "Passed all checks",
        "x_range": x_diff,
        "y_range": y_diff,
        "unique_ratio": unique_ratio,
        "aspect_ratio": aspect_ratio,
    }


def search_attractors(
    equation_id: str,
    num_to_find: int = 10,
    max_attempts: int = 5_000,
    parameter_ranges: Dict[str, Tuple[float, float]] = None,
    test_iterations: int = 25_000,
    final_iterations: int = ITERATIONS_DEFAULT,
    decimals: int = 2,
    output_dir: str = None,
    prefix: str = "attractor",
    progress_interval: int = 50,
    start_counter: int = 1,
    min_small_side: float = 0.25,
    max_small_side: float = 500.0,
    digits_unique: int = 2,
    min_unique_ratio: float = 0.1,
    max_unique_ratio: float = 0.95,
    max_aspect_ratio: float = 4.0,
) -> Dict[str, Any]:
    """
    Perform random search over parameter space to find interesting attractors.
    
    Randomly samples parameter combinations, evaluates them with quick tests,
    then generates and saves full-resolution visualizations of those that pass.
    Creates both individual PDF files and a summary CSV.
    
    Args:
        equation_id: Name of equation system from EQUATION_LIBRARY
        num_to_find: Target number of attractors to discover
        max_attempts: Maximum parameter sets to test before stopping
        parameter_ranges: Dict mapping 'a','b','c','d' to (min, max) tuples
        test_iterations: Points to generate during quick evaluation
        final_iterations: Points to generate for saved visualizations
        decimals: Decimal places for parameter sampling
        output_dir: Directory for saving results (created if needed)
        prefix: Filename prefix for saved attractors
        progress_interval: Print status every N attempts
        start_counter: Starting number for attractor naming
        min_small_side: Minimum range in either dimension (evaluation)
        max_small_side: Maximum range in either dimension (evaluation)
        digits_unique: Decimal places for uniqueness calculation
        min_unique_ratio: Minimum fraction of unique points (evaluation)
        max_unique_ratio: Maximum fraction of unique points (evaluation)
        max_aspect_ratio: Maximum ratio of dimensions (evaluation)
        
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

    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    found_attractors = []
    attempts = 0

    print(f"Starting search for {num_to_find} attractors (equation: {equation_id})...")
    start_time = time.time()

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

        evaluation = evaluate_attractor(
            params,
            equation_id=equation_id,
            test_iterations=test_iterations,
            min_small_side=min_small_side,
            max_small_side=max_small_side,
            digits_unique=digits_unique,
            min_unique_ratio=min_unique_ratio,
            max_unique_ratio=max_unique_ratio,
            max_aspect_ratio=max_aspect_ratio,
        )

        if attempts % progress_interval == 0:
            elapsed_sec = time.time() - start_time
            rate = elapsed_sec / attempts if attempts > 0 else 0.0
            print(
                f"\rAttempts: {attempts} | "
                f"Found: {len(found_attractors)}/{num_to_find} | "
                f"Elapsed: {elapsed_sec:.1f} sec | Rate: {rate:.2f} sec/test",
                end="",
                flush=True,
            )

        # Skip if evaluation failed
        if evaluation["score"] < 0:
            continue

        # Passed quick checks; now do full iterations and save
        attractor_num = len(found_attractors) + start_counter
        print(
            f"\n\nFound candidate {attractor_num} at attempt {attempts} "
            f"with params: a={params['a']}, b={params['b']}, "
            f"c={params['c']}, d={params['d']}"
        )

        try:
            data = prepare_attractor_data(
                params,
                equation_id=equation_id,
                n_iterations=final_iterations,
            )
        except Exception as e:
            print(f"Skipped attractor {attractor_num} due to data prep error: {e}")
            continue

        x = data["x"]
        y = data["y"]

        # Remove any NaN or Inf values
        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]

        x_min = float(np.min(x))
        x_max = float(np.max(x))
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        x_diff = x_max - x_min
        y_diff = y_max - y_min

        # Re-apply geometric checks on full-resolution data
        if x_diff < min_small_side or y_diff < min_small_side:
            print(
                f"Skipped attractor {attractor_num} due to collapsed range: "
                f"x_diff={x_diff:.3f}, y_diff={y_diff:.3f}"
            )
            continue
        
        if x_diff > max_small_side or y_diff > max_small_side:
            print(
                f"Skipped attractor {attractor_num} due to divergent behavior: "
                f"x_diff={x_diff:.3f}, y_diff={y_diff:.3f}"
            )
            continue

        rounded_points = np.column_stack(
            [np.round(x, digits_unique), np.round(y, digits_unique)]
        )
        unique_points = np.unique(rounded_points, axis=0)
        unique_ratio = unique_points.shape[0] / rounded_points.shape[0]

        if unique_ratio < min_unique_ratio:
            print(
                f"Skipped attractor {attractor_num} due to low unique ratio: "
                f"{unique_ratio:.4f}"
            )
            continue

        if unique_ratio > max_unique_ratio:
            print(
                f"Skipped attractor {attractor_num} due to high unique ratio: "
                f"{unique_ratio:.4f}"
            )
            continue

        aspect_ratio = x_diff / y_diff
        max_aspect_component = max(aspect_ratio, 1.0 / aspect_ratio)

        if max_aspect_component > max_aspect_ratio:
            print(
                f"Skipped attractor {attractor_num} due to bad aspect ratio: "
                f"{max_aspect_component:.2f}"
            )
            continue

        # Scoring (lower is better)
        ideal_aspect_ratio = 1.5
        minimum_aspect_ratio = 1.0
        aspect_normalized = (max_aspect_component - ideal_aspect_ratio) / (max_aspect_ratio - minimum_aspect_ratio)
        
        ideal_unique_ratio = 0.75
        unique_normalized = (unique_ratio - ideal_unique_ratio) / (max_unique_ratio - min_unique_ratio)
        
        score_final = aspect_normalized**2 + unique_normalized**2

        print(
            f"Score: {score_final:.2f} | "
            f"X range: {x_diff:.3f} | Y range: {y_diff:.3f} | "
            f"Aspect ratio: {aspect_ratio:.2f} | "
            f"Unique ratio: {unique_ratio:.3f}"
        )

        # Save attractor image with equation panel as a PDF in output_dir
        base_filename = f"{prefix}_{attractor_num}"
        output_path = os.path.join(output_dir, base_filename)

        saved_files = save_attractor(
            data,
            output_path,
            save_format="pdf",
            include_info=True,
            equation_id=equation_id,
            color_method=COLOR_METHOD,
            palette=VIRIDIS_PALETTE,
            low=GRADIENT_LOW,
            high=GRADIENT_HIGH,
        )

        found_attractors.append(
            {
                "parameters": params,
                "evaluation": evaluation,
                "final_stats": {
                    "score": score_final,
                    "x_diff": x_diff,
                    "y_diff": y_diff,
                    "aspect_ratio": aspect_ratio,
                    "unique_ratio": unique_ratio,
                },
                "files": saved_files,
            }
        )

    elapsed_minutes = (time.time() - start_time) / 60.0
    print(
        f"\n\nSearch completed after {attempts} attempts "
        f"({elapsed_minutes:.2f} minutes)"
    )
    print(f"Found {len(found_attractors)} attractors")

    # Build summary table
    summary_rows = []
    for index, attr in enumerate(found_attractors, start=start_counter):
        row = {
            "name": f"{prefix}_{index}",
            "a": attr["parameters"]["a"],
            "b": attr["parameters"]["b"],
            "c": attr["parameters"]["c"],
            "d": attr["parameters"]["d"],
            "score": attr["final_stats"]["score"],
            "x_range": attr["final_stats"]["x_diff"],
            "y_range": attr["final_stats"]["y_diff"],
            "aspect_ratio": attr["final_stats"]["aspect_ratio"],
            "unique_ratio": attr["final_stats"]["unique_ratio"],
        }
        summary_rows.append(row)

    # Write CSV summary if any results
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

    return {
        "attractors": found_attractors,
        "summary": summary_rows,
        "attempts": attempts,
        "elapsed_minutes": elapsed_minutes,
    }
