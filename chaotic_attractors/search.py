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
from scipy.stats import gaussian_kde


from .core import (
    COLOR_METHOD,
    VIRIDIS_PALETTE,
    GRADIENT_LOW,
    GRADIENT_HIGH,
    generate_chaotic,
    save_attractor,
    evaluate_attractor
)
from .equations import EQUATION_LIBRARY


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
    test_iterations: int = 25_000,
    final_iterations: int = 2_000_000,
    decimals: int = 2,
    output_dir: str = "output",
    prefix: str | None = None,
    progress_interval: int = 50,
    start_counter: int = 1,
    min_small_side: float = 0.25,
    max_small_side: float = 500.0,
    digits_unique: int = 2,
    min_unique_ratio: float = 0.1,
    max_unique_ratio: float = 0.95,
    max_aspect_ratio: float = 4.0,
    include_info: bool = True,
    save_format: str = 'png',
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
    
            evaluation = evaluate_attractor(
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
                    f"Elapsed: {elapsed_sec:.1f} sec | Rate: {rate:.2f} sec/test",
                    end="",
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
            
            try:
                data = prepare_search_data(
                    params=params,
                    x_start=x_start,
                    y_start=y_start,
                    equation_id=equation_id,
                    final_iterations=final_iterations,
                )
            except Exception as e:
                print(f"Skipped attractor {attractor_num} due to data prep error: {e}")
                continue
    
            evaluation = evaluate_attractor(
                params=params,
                equation_id=equation_id,
                x=data["x"],
                y=data["y"],
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
                continue
            
            # Passed final checks
            print("\nFinal tests passed")
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
            f"\n\nSearch completed after {attempts} attempts "
            f"({elapsed_minutes:.2f} minutes)"
        )
        print(f"Found {len(found_attractors)} attractors")
    
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
    
        return {
            "attractors": found_attractors,
            "summary": summary_rows,
            "attempts": attempts,
            "elapsed_minutes": elapsed_minutes,
        }


def prepare_search_data(
    params: Dict[str, float],
    x_start: float,
    y_start: float,
    equation_id: str,
    final_iterations: int = 2_000_000,
    kde_sample_size: int = 50_000,
) -> Dict[str, Any]:
    """
    Generate attractor points and compute density using KDE for visualization.
    
    Returns dict with keys: 'x', 'y', 'density' (normalized 0-1), 'params'.
    Raises ValueError if fewer than 10,000 valid points are generated.
    """
    
    # Generate points
    x, y = generate_chaotic(
        params=params,
        equation_id=equation_id,
        iterations=final_iterations,
        x_start=x_start,
        y_start=y_start)

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
            print(f"Normalized Density Range: {density.min():.2f} to {density.max():.2f}")
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
        'params': params,
        'equation_id': equation_id
    }
