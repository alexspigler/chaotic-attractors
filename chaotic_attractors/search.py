"""Parameter search and validation for four-parameter attractors."""

import csv
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .core import (
    COLOR_METHOD,
    GRADIENT_HIGH,
    GRADIENT_LOW,
    VIRIDIS_PALETTE,
    _compute_density,
    _screen_geometry,
    evaluate_attractor_first,
    generate_chaotic,
    save_attractor,
)
from .metrics import estimate_largest_lyapunov


def generate_random(
    min_val: float,
    max_val: float,
    decimals: int,
    rng: np.random.Generator | None = None,
) -> float:
    """Sample uniformly from the decimal grid inside the requested bounds."""
    if not np.isfinite(min_val) or not np.isfinite(max_val):
        raise ValueError("range bounds must be finite")
    if min_val >= max_val:
        raise ValueError(f"Invalid range: min_val={min_val}, max_val={max_val}")
    if decimals < 0 or decimals > 8:
        raise ValueError("decimals must be between 0 and 8")

    scale = 10**decimals
    lower_index = math.ceil(min_val * scale - 1e-12)
    upper_index = math.floor(max_val * scale + 1e-12)
    if lower_index > upper_index:
        raise ValueError("the requested range contains no values at this precision")

    rng = np.random.default_rng() if rng is None else rng
    index = int(rng.integers(lower_index, upper_index + 1))
    return index / scale


def evaluate_attractor_second(
    params: Mapping[str, float],
    equation_id: str,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    x_start: float | None = None,
    y_start: float | None = None,
    min_small_side: float = 0.25,
    max_small_side: float = 500.0,
    digits_unique: int = 4,
    min_unique_ratio: float = 0.25,
    max_unique_ratio: float = 1.0,
    max_aspect_ratio: float = 4.0,
    iterations: int | None = None,
    burn_in: int = 0,
    lyapunov_iterations: int = 10_000,
    min_lyapunov_exponent: float = 0.0,
) -> dict[str, Any]:
    """Apply full geometric, recurrence, and sensitivity checks."""
    if digits_unique < 0:
        raise ValueError("digits_unique cannot be negative")
    if not 0.0 <= min_unique_ratio < max_unique_ratio <= 1.0:
        raise ValueError("unique-ratio bounds must satisfy 0 <= min < max <= 1")
    if max_aspect_ratio < 1.0:
        raise ValueError("max_aspect_ratio must be at least 1")

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

    x_values = np.asarray(screen["x"])
    y_values = np.asarray(screen["y"])
    x_range = float(screen["x_diff"])
    y_range = float(screen["y_diff"])

    rounded_points = np.column_stack(
        [np.round(x_values, digits_unique), np.round(y_values, digits_unique)]
    )
    unique_ratio = np.unique(rounded_points, axis=0).shape[0] / len(rounded_points)
    if unique_ratio < min_unique_ratio:
        return {
            "score": -1.0,
            "reason": f"Low unique point ratio ({unique_ratio:.4f}) - likely collapses",
        }
    if unique_ratio > max_unique_ratio:
        return {
            "score": -1.0,
            "reason": f"High unique point ratio ({unique_ratio:.4f})",
        }

    aspect_ratio = max(x_range, y_range) / min(x_range, y_range)
    if aspect_ratio > max_aspect_ratio:
        return {
            "score": -1.0,
            "reason": f"Bad aspect ratio ({aspect_ratio:.2f})",
        }

    lyapunov_exponent = estimate_largest_lyapunov(
        x=x_values,
        y=y_values,
        params=params,
        equation_id=equation_id,
        max_iterations=lyapunov_iterations,
    )
    if not np.isfinite(lyapunov_exponent) or lyapunov_exponent <= min_lyapunov_exponent:
        return {
            "score": -1.0,
            "reason": (
                "Finite-time Lyapunov exponent "
                f"{lyapunov_exponent:.4f} is not above "
                f"{min_lyapunov_exponent:.4f}"
            ),
            "lyapunov_exponent": lyapunov_exponent,
        }

    ideal_aspect_ratio = 1.5
    ideal_unique_ratio = 2.0 / 3.0
    max_aspect_deviation = max(
        abs(1.0 - ideal_aspect_ratio),
        abs(max_aspect_ratio - ideal_aspect_ratio),
    )
    max_unique_deviation = max(
        abs(min_unique_ratio - ideal_unique_ratio),
        abs(max_unique_ratio - ideal_unique_ratio),
    )
    aspect_normalized = (aspect_ratio - ideal_aspect_ratio) / max_aspect_deviation
    unique_normalized = (unique_ratio - ideal_unique_ratio) / max_unique_deviation

    return {
        "score": aspect_normalized**2 + unique_normalized**2,
        "reason": "Passed all checks",
        "x_range": x_range,
        "y_range": y_range,
        "unique_ratio": unique_ratio,
        "aspect_ratio": aspect_ratio,
        "lyapunov_exponent": lyapunov_exponent,
    }


def prepare_search_data(
    params: Mapping[str, float],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    equation_id: str,
    kde_sample_size: int = 50_000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Compute point density for a validated search result."""
    density = _compute_density(x, y, kde_sample_size, rng=rng)
    return {
        "x": x,
        "y": y,
        "density": density,
        "params": dict(params),
        "equation_id": equation_id,
    }


def _validate_search_settings(
    num_to_find: int,
    max_attempts: int,
    test_iterations: int,
    final_iterations: int,
    progress_interval: int,
    parameter_ranges: Mapping[str, tuple[float, float]],
) -> None:
    if num_to_find <= 0:
        raise ValueError("num_to_find must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if test_iterations <= 0:
        raise ValueError("test_iterations must be positive")
    if final_iterations < 10_000:
        raise ValueError("final_iterations must be at least 10,000")
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")

    missing = {"a", "b", "c", "d"} - parameter_ranges.keys()
    if missing:
        raise KeyError(f"Missing parameter ranges: {missing}")
    for name in ("a", "b", "c", "d"):
        lower, upper = parameter_ranges[name]
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"Invalid range for {name}: ({lower}, {upper})")


def search_attractors(
    equation_id: str,
    x_start: float,
    y_start: float,
    num_to_find: int = 10,
    max_attempts: int = 5_000,
    parameter_ranges: Mapping[str, tuple[float, float]] | None = None,
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
    burn_in: int = 1_000,
    lyapunov_iterations: int = 10_000,
    min_lyapunov_exponent: float = 0.0,
    kde_sample_size: int = 50_000,
    seed: int | None = None,
    overwrite: bool = False,
    rasterized_points: bool = True,
) -> dict[str, Any]:
    """Search a discrete parameter space and save candidates that pass validation."""
    if parameter_ranges is None:
        parameter_ranges = {
            "a": (-3.0, 3.0),
            "b": (-3.0, 3.0),
            "c": (-3.0, 3.0),
            "d": (-3.0, 3.0),
        }

    _validate_search_settings(
        num_to_find=num_to_find,
        max_attempts=max_attempts,
        test_iterations=test_iterations,
        final_iterations=final_iterations,
        progress_interval=progress_interval,
        parameter_ranges=parameter_ranges,
    )
    if burn_in < 0:
        raise ValueError("burn_in cannot be negative")
    if start_counter < 1:
        raise ValueError("start_counter must be at least 1")
    if kde_sample_size < 2:
        raise ValueError("kde_sample_size must be at least 2")

    prefix = equation_id if prefix is None else prefix
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / f"{prefix}_summary.csv"
    if csv_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {csv_path}. "
            "Pass overwrite=True to replace it."
        )

    seed_sequence = np.random.SeedSequence(seed)
    effective_seed = int(seed_sequence.entropy)
    rng = np.random.default_rng(seed_sequence)

    found_attractors: list[dict[str, Any]] = []
    seen_parameters: set[tuple[float, float, float, float]] = set()
    attempts = 0
    duplicates_skipped = 0
    print(f"\nStarting search for {num_to_find} attractors ({equation_id})...")
    print(f"Random seed: {effective_seed}")
    start_time = time.monotonic()

    try:
        while len(found_attractors) < num_to_find and attempts < max_attempts:
            attempts += 1
            params = {
                name: generate_random(*parameter_ranges[name], decimals, rng=rng)
                for name in ("a", "b", "c", "d")
            }
            parameter_key = tuple(params[name] for name in ("a", "b", "c", "d"))
            if parameter_key in seen_parameters:
                duplicates_skipped += 1
                continue
            seen_parameters.add(parameter_key)

            evaluation = evaluate_attractor_first(
                params=params,
                x_start=x_start,
                y_start=y_start,
                equation_id=equation_id,
                iterations=test_iterations,
                min_small_side=min_small_side,
                max_small_side=max_small_side,
                max_aspect_ratio=max_aspect_ratio,
                burn_in=burn_in,
            )

            if attempts % progress_interval == 0:
                elapsed_seconds = time.monotonic() - start_time
                tests_per_second = (
                    attempts / elapsed_seconds if elapsed_seconds else 0.0
                )
                print(
                    f"\rAttempts: {attempts} | "
                    f"Found: {len(found_attractors)}/{num_to_find} | "
                    f"Elapsed: {elapsed_seconds:.1f} sec | "
                    f"Rate: {tests_per_second:.2f} test/sec",
                    end="",
                    flush=True,
                )

            if evaluation["score"] < 0:
                continue

            attractor_number = len(found_attractors) + start_counter
            print(
                f"\nCandidate {attractor_number} passed the short screen at "
                f"attempt {attempts}: {params}"
            )
            x, y = generate_chaotic(
                params=params,
                equation_id=equation_id,
                iterations=final_iterations,
                x_start=x_start,
                y_start=y_start,
                burn_in=burn_in,
            )
            if len(x) != final_iterations:
                print(
                    "Full trajectory terminated early: "
                    f"{len(x):,} of {final_iterations:,} points"
                )
                continue

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
                lyapunov_iterations=lyapunov_iterations,
                min_lyapunov_exponent=min_lyapunov_exponent,
            )
            if evaluation["score"] < 0:
                print(f"Full validation failed: {evaluation['reason']}")
                continue

            data = prepare_search_data(
                params=params,
                x=x,
                y=y,
                kde_sample_size=kde_sample_size,
                equation_id=equation_id,
                rng=rng,
            )
            data.update(
                {
                    "burn_in": burn_in,
                    "lyapunov_exponent": evaluation["lyapunov_exponent"],
                    "seed": effective_seed,
                }
            )
            saved_files = save_attractor(
                data=data,
                x_start=x_start,
                y_start=y_start,
                output_dir=str(output_directory),
                prefix=prefix,
                start_counter=attractor_number,
                save_format=save_format,
                include_info=include_info,
                color_method=COLOR_METHOD,
                palette=VIRIDIS_PALETTE,
                low=GRADIENT_LOW,
                high=GRADIENT_HIGH,
                overwrite=overwrite,
                rasterized_points=rasterized_points,
            )
            found_attractors.append(
                {
                    "parameters": params,
                    "evaluation": evaluation,
                    "files": saved_files,
                }
            )
            print(
                f"Accepted with Lyapunov exponent "
                f"{evaluation['lyapunov_exponent']:.4f}"
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    elapsed_minutes = (time.monotonic() - start_time) / 60.0
    rate = len(found_attractors) / elapsed_minutes if elapsed_minutes else 0.0
    print(
        f"\n{len(found_attractors)} attractors completed after {attempts} attempts "
        f"({elapsed_minutes:.2f} minutes; {rate:.2f} attractors/minute)"
    )

    summary_rows = []
    for index, attractor in enumerate(found_attractors, start=start_counter):
        evaluation = attractor["evaluation"]
        summary_rows.append(
            {
                "name": f"{prefix}_{index}",
                **attractor["parameters"],
                "score": evaluation["score"],
                "x_range": evaluation["x_range"],
                "y_range": evaluation["y_range"],
                "aspect_ratio": evaluation["aspect_ratio"],
                "unique_ratio": evaluation["unique_ratio"],
                "lyapunov_exponent": evaluation["lyapunov_exponent"],
                "burn_in": burn_in,
                "seed": effective_seed,
            }
        )
    summary_rows.sort(key=lambda row: row["score"])

    if summary_rows:
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
            "lyapunov_exponent",
            "burn_in",
            "seed",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved parameter summary to: {csv_path}")
    else:
        print("No attractors found that met the criteria.")

    return {
        "attractors": found_attractors,
        "summary": summary_rows,
        "attempts": attempts,
        "duplicates_skipped": duplicates_skipped,
        "elapsed_minutes": elapsed_minutes,
        "seed": effective_seed,
    }
