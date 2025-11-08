"""
Command-line entry point for the chaotic attractor visualizer.

Modes:
- run:    Generate a single attractor with given (or default) parameters
- search: Run automated parameter search over a,b,c,d in a shared range
"""

import argparse
import sys
from typing import Dict

from .equations import EQUATION_LIBRARY
from .core import (
    ITERATIONS_DEFAULT,
    COLOR_METHOD,
    VIRIDIS_PALETTE,
    GRADIENT_LOW,
    GRADIENT_HIGH,
    prepare_attractor_data,
    save_attractor,
)
from .search import search_attractors


# Default parameters for "run" mode
DEFAULT_EQUATION = "Tinkerbell"
DEFAULT_PARAMS = {
    "a": 0.9,
    "b": -0.6013,
    "c": 2.0,
    "d": 0.5,
}

# Default parameter range for "search" mode (same for a,b,c,d)
DEFAULT_RANGE_MIN = -3.0
DEFAULT_RANGE_MAX = 3.0
DEFAULT_DECIMALS = 2


def validate_equation(equation_id: str) -> None:
    """
    Check if equation exists in library, exit with error if not.
    
    Args:
        equation_id: Name of equation to validate
        
    Raises:
        SystemExit: If equation_id not found in EQUATION_LIBRARY
    """
    if equation_id not in EQUATION_LIBRARY:
        available = ", ".join(EQUATION_LIBRARY.keys())
        print(f"Error: Unknown equation '{equation_id}'")
        print(f"Available equations: {available}")
        sys.exit(1)


def run_single_mode(equation_id: str, params: Dict[str, float]) -> None:
    """
    Generate a single attractor and save both standard and annotated versions.
    
    Creates two outputs:
    1. Standard visualization without equation panel
    2. Annotated version with equation and parameters displayed
    
    Args:
        equation_id: Name of equation system from EQUATION_LIBRARY
        params: Dictionary with keys 'a', 'b', 'c', 'd'
    """
    validate_equation(equation_id)
    
    print(f"Generating attractor: {equation_id}")
    print(
        f"Parameters: a={params['a']}, b={params['b']}, "
        f"c={params['c']}, d={params['d']}"
    )
    print(f"Iterations: {ITERATIONS_DEFAULT:,}")

    # Generate data
    data = prepare_attractor_data(params, equation_id)

    print(f"Generated {len(data['x']):,} valid points")
    print(f"X range: [{data['x'].min():.3f}, {data['x'].max():.3f}]")
    print(f"Y range: [{data['y'].min():.3f}, {data['y'].max():.3f}]")

    # Save standard version (without equation)
    print("\nSaving standard version...")
    save_attractor(
        data,
        filename="attractor_output",
        save_format="png",  # png, pdf, svg, all
        color_method=COLOR_METHOD,
        palette=VIRIDIS_PALETTE,
        low=GRADIENT_LOW,
        high=GRADIENT_HIGH,
    )

    # Save version with equation panel
    print("\nSaving version with equation...")
    save_attractor(
        data,
        filename="attractor_with_eq",
        save_format="all",
        include_info=True,
        equation_id=equation_id,
        color_method=COLOR_METHOD,
        palette=VIRIDIS_PALETTE,
        low=GRADIENT_LOW,
        high=GRADIENT_HIGH,
    )

    print("\nDone!")


def run_search_mode(
    equation_id: str,
    decimals: int,
    range_min: float,
    range_max: float,
) -> None:
    """
    Search parameter space for interesting attractors.
    
    Randomly samples parameter combinations within specified ranges,
    evaluates candidates using geometric and statistical criteria,
    and saves high-quality attractors as PDFs with a summary CSV.
    Uses a shared [range_min, range_max] for all four parameters.
    
    Args:
        equation_id: Name of equation system from EQUATION_LIBRARY
        decimals: Number of decimal places for parameter sampling
        range_min: Lower bound for all parameters
        range_max: Upper bound for all parameters
    """
    validate_equation(equation_id)
    
    print(f"Running search mode for equation: {equation_id}")
    print(
        f"Parameter ranges: a,b,c,d in [{range_min}, {range_max}] "
        f"with {decimals} decimal places"
    )

    parameter_ranges = {
        "a": (range_min, range_max),
        "b": (range_min, range_max),
        "c": (range_min, range_max),
        "d": (range_min, range_max),
    }

    results = search_attractors(
        equation_id=equation_id,
        num_to_find=15,
        max_attempts=20_000,
        parameter_ranges=parameter_ranges,
        test_iterations=10_000,
        final_iterations=ITERATIONS_DEFAULT,
        decimals=decimals,
        output_dir="search_results",
        prefix=equation_id,
        start_counter=1,
        min_small_side=0.25,
        max_small_side=500.0,
        digits_unique=4,
        min_unique_ratio=0.25,
        max_unique_ratio=1.1,
        max_aspect_ratio=4.0,
    )

    print("\nSearch complete.")
    print(
        f"Attempts: {results['attempts']} | "
        f"Found: {len(results['attractors'])} | "
        f"Elapsed: {results['elapsed_minutes']:.2f} minutes"
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for command-line interface.
    
    Returns:
        Configured ArgumentParser with all CLI options
    """
    parser = argparse.ArgumentParser(
        description="4-Parameter Chaotic Attractor Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available equations: {', '.join(EQUATION_LIBRARY.keys())}

Examples:
  # Generate single attractor with default parameters
  python -m chaotic_attractors.main
  
  # Generate with custom parameters
  python -m chaotic_attractors.main --a 0.9 --b -0.6 --c 2.0 --d 0.5
  
  # Search for attractors
  python -m chaotic_attractors.main --mode search --equation Clifford
  
  # Search with custom ranges
  python -m chaotic_attractors.main --mode search --range-min -2 --range-max 2
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["run", "search"],
        default="run",
        help="Mode to run: 'run' (single attractor) or 'search' (parameter search)",
    )

    parser.add_argument(
        "--equation",
        default=DEFAULT_EQUATION,
        help=f"Equation ID from the equation library (default: {DEFAULT_EQUATION})",
    )

    # Parameters for run mode
    parser.add_argument("--a", type=float, help="Parameter a for run mode")
    parser.add_argument("--b", type=float, help="Parameter b for run mode")
    parser.add_argument("--c", type=float, help="Parameter c for run mode")
    parser.add_argument("--d", type=float, help="Parameter d for run mode")

    # Parameters for search mode
    parser.add_argument(
        "--decimals",
        type=int,
        default=DEFAULT_DECIMALS,
        help=f"Number of decimal places for parameter sampling in search mode "
             f"(default: {DEFAULT_DECIMALS})",
    )
    parser.add_argument(
        "--range-min",
        type=float,
        default=DEFAULT_RANGE_MIN,
        help=f"Lower bound for a,b,c,d in search mode (default: {DEFAULT_RANGE_MIN})",
    )
    parser.add_argument(
        "--range-max",
        type=float,
        default=DEFAULT_RANGE_MAX,
        help=f"Upper bound for a,b,c,d in search mode (default: {DEFAULT_RANGE_MAX})",
    )

    return parser


def main() -> None:
    """
    Parse command-line arguments and execute requested mode.
    
    Entry point for the command-line interface. Dispatches to either
    run_single_mode or run_search_mode based on --mode flag.
    """
    parser = build_parser()
    args = parser.parse_args()

    equation_id = args.equation

    try:
        if args.mode == "run":
            # Start from defaults and override if flags are provided
            params = DEFAULT_PARAMS.copy()
            if args.a is not None:
                params["a"] = args.a
            if args.b is not None:
                params["b"] = args.b
            if args.c is not None:
                params["c"] = args.c
            if args.d is not None:
                params["d"] = args.d
    
            run_single_mode(equation_id, params)

        elif args.mode == "search":
            run_search_mode(
                equation_id=equation_id,
                decimals=args.decimals,
                range_min=args.range_min,
                range_max=args.range_max,
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(130)

if __name__ == "__main__":
    main()
