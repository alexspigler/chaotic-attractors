"""
Command-line entry point for the chaotic attractor visualizer.

Modes:
- generate: Generate a single attractor with given (or default) parameters
- search: Run automated parameter search over a,b,c,d in specified ranges
"""

import argparse
import sys
import textwrap
import warnings

from .core import (
    prepare_generate_data,
    save_attractor,
)
from .equations import EQUATION_LIBRARY
from .search import (
    search_attractors,
)

# Default parameters for "generate" mode
DEFAULT_EQUATION = "Tinkerbell"
DEFAULT_PARAMS = {
    "a": 0.9,
    "b": -0.6013,
    "c": 2.0,
    "d": 0.5,
}
DEFAULT_TEST_ITERATIONS = 25_000
DEFAULT_FINAL_ITERATIONS = 2_000_000
DEFAULT_X_START = 0.0
DEFAULT_Y_START = 0.0

DEFAULT_FORMAT = "png"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_INCLUDE_INFO = True

# Default parameter range for "search" mode
DEFAULT_RANGE_MIN = -3.0
DEFAULT_RANGE_MAX = 3.0
DEFAULT_DECIMALS = 2
DEFAULT_NUMBER = 5
DEFAULT_MAX_ATTEMPTS = 20_000
DEFAULT_TEST_ITERATIONS = 10_000
DEFAULT_FINAL_ITERATIONS = 2_000_000


def build_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for command-line interface.

    Returns:
        Configured ArgumentParser with all CLI options
    """
    epilog = textwrap.dedent(
        """\
        Examples:
          # Generate Mode
          chaotic_attractors \\
              --equation Custom3 \\
              --a -2.17 --b -2.7 --c -2.08 --d -2.83 \\
              --x-start 0 \\
              --y-start 0 \\
              --test-iter 50000 \\
              --final-iter 2000000 \\
              --format png \\
              --output-dir output \\
              --info-panel

          # Search Mode
          chaotic_attractors \\
              --mode search \\
              --equation Custom3 \\
              --range-min -3 \\
              --range-max 3 \\
              --x-start 0.5 \\
              --y-start 0.5 \\
              --decimals 2 \\
              --num-to-find 10 \\
              --max-attempts 50000 \\
              --test-iter 50000 \\
              --final-iter 2000000 \\
              --format png \\
              --output-dir output \\
              --info-panel
        """
    )

    parser = argparse.ArgumentParser(
        description="Chaotic Attractors: Computational Exploration & Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )

    parser.add_argument(
        "--mode",
        choices=["generate", "search"],
        default="generate",
        help="Mode to run: 'generate' (single attractor) or 'search' (attractor search)",
    )

    parser.add_argument(
        "--equation",
        choices=EQUATION_LIBRARY.keys(),
        default=DEFAULT_EQUATION,
        help=f"Equation ID from the equation library (default: {DEFAULT_EQUATION})",
    )

    parser.add_argument(
        "--a", type=float, metavar="FLOAT", help="Parameter a for generate mode (float)"
    )
    parser.add_argument(
        "--b", type=float, metavar="FLOAT", help="Parameter b for generate mode (float)"
    )
    parser.add_argument(
        "--c", type=float, metavar="FLOAT", help="Parameter c for generate mode (float)"
    )
    parser.add_argument(
        "--d", type=float, metavar="FLOAT", help="Parameter d for generate mode (float)"
    )

    parser.add_argument(
        "--test-iter",
        type=int,
        metavar="INT",
        help="Number of iterations for initial tests (int)",
    )
    parser.add_argument(
        "--final-iter", type=int, metavar="INT", help="Number of final iterations (int)"
    )

    parser.add_argument(
        "--x-start", type=float, metavar="FLOAT", help="Starting x-value (float)"
    )
    parser.add_argument(
        "--y-start", type=float, metavar="FLOAT", help="Starting y-value (float)"
    )

    parser.add_argument(
        "--format",
        choices=["all", "pdf", "png", "svg"],
        default="png",
        help="Choose which format you'd like to save as",
    )

    parser.add_argument(
        "--info-panel",
        action="store_true",
        help="If you want equation and parameter info at bottom of attractor",
    )

    parser.add_argument(
        "--range-min",
        type=float,
        metavar="FLOAT",
        default=DEFAULT_RANGE_MIN,
        help=f"Lower bound for a,b,c,d in search mode (float, default: {DEFAULT_RANGE_MIN})",
    )

    parser.add_argument(
        "--range-max",
        type=float,
        metavar="FLOAT",
        default=DEFAULT_RANGE_MAX,
        help=f"Upper bound for a,b,c,d in search mode (float, default: {DEFAULT_RANGE_MAX})",
    )

    parser.add_argument(
        "--decimals",
        type=int,
        metavar="INT",
        default=DEFAULT_DECIMALS,
        help=f"Number of decimal places for parameter sampling in search mode "
        f"(int, default: {DEFAULT_DECIMALS})",
    )

    parser.add_argument(
        "--num-to-find",
        type=int,
        metavar="INT",
        default=DEFAULT_NUMBER,
        help="Number of attractors to find before stopping search (int)",
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        metavar="INT",
        default=DEFAULT_MAX_ATTEMPTS,
        help="Number of attempts at which search will stop, even if desired number of attractors not found (int)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        metavar="PATH",
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for all found attractors when in search mode (path)",
    )

    return parser


def main() -> None:
    """
    Entry point for the command-line interface.
    """
    # Suppress runtime warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.mode == "generate":
            # Start from defaults and override if flags are provided
            equation_id = args.equation or DEFAULT_EQUATION

            params = DEFAULT_PARAMS
            if args.a is not None:
                params["a"] = args.a
            if args.b is not None:
                params["b"] = args.b
            if args.c is not None:
                params["c"] = args.c
            if args.d is not None:
                params["d"] = args.d

            test_iterations = args.test_iter or DEFAULT_TEST_ITERATIONS
            final_iterations = args.final_iter or DEFAULT_FINAL_ITERATIONS

            x_start = args.x_start or DEFAULT_X_START
            y_start = args.y_start or DEFAULT_Y_START

            save_format = args.format or DEFAULT_FORMAT
            output_dir = args.output_dir or DEFAULT_OUTPUT_DIR

            data = prepare_generate_data(
                params=params,
                equation_id=equation_id,
                test_iterations=test_iterations,
                final_iterations=final_iterations,
                x_start=x_start,
                y_start=y_start,
            )

            save_attractor(
                data=data,
                x_start=x_start,
                y_start=y_start,
                save_format=save_format,
                output_dir=output_dir,
                include_info=args.info_panel,
            )

        elif args.mode == "search":
            # Start from defaults and override if flags are provided
            equation_id = args.equation or DEFAULT_EQUATION

            num_to_find = args.num_to_find or DEFAULT_NUMBER
            max_attempts = args.max_attempts or DEFAULT_MAX_ATTEMPTS
            decimals = args.decimals or DEFAULT_DECIMALS
            parameter_ranges = {
                "a": (args.range_min, args.range_max),
                "b": (args.range_min, args.range_max),
                "c": (args.range_min, args.range_max),
                "d": (args.range_min, args.range_max),
            }
            x_start = args.x_start or DEFAULT_X_START
            y_start = args.y_start or DEFAULT_Y_START
            test_iterations = args.test_iter or DEFAULT_TEST_ITERATIONS
            final_iterations = args.final_iter or DEFAULT_FINAL_ITERATIONS
            save_format = args.format or DEFAULT_FORMAT

            search_attractors(
                equation_id=equation_id,
                num_to_find=num_to_find,
                max_attempts=max_attempts,
                parameter_ranges=parameter_ranges,
                test_iterations=test_iterations,
                final_iterations=final_iterations,
                x_start=x_start,
                y_start=y_start,
                decimals=decimals,
                include_info=args.info_panel,
                save_format=save_format,
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(130)


if __name__ == "__main__":
    main()
