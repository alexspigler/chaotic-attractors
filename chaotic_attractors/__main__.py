"""Command-line interface for chaotic-attractors."""

import argparse
import sys
import textwrap
import warnings

from .core import prepare_generate_data, save_attractor
from .equations import EQUATION_LIBRARY
from .search import search_attractors

DEFAULT_EQUATION = "Tinkerbell"
DEFAULT_PARAMS = {"a": 0.9, "b": -0.6013, "c": 2.0, "d": 0.5}
DEFAULT_STARTS = {
    "Tinkerbell": (-0.72, -0.64),
    "Custom2": (0.0, 0.0),
    "Custom3": (0.0, 0.0),
}
DEFAULT_SEARCH_START = (0.5, 0.5)
DEFAULT_TEST_ITERATIONS = 100_000
DEFAULT_FINAL_ITERATIONS = 2_000_000
DEFAULT_BURN_IN = 1_000
DEFAULT_LYAPUNOV_ITERATIONS = 10_000
DEFAULT_KDE_SAMPLE_SIZE = 50_000
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_RANGE_MIN = -3.0
DEFAULT_RANGE_MAX = 3.0
DEFAULT_DECIMALS = 2
DEFAULT_NUMBER = 5
DEFAULT_MAX_ATTEMPTS = 20_000
DEFAULT_PROGRESS_INTERVAL = 50


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    epilog = textwrap.dedent("""\
        Examples:
          chaotic-attractors --equation Tinkerbell --seed 42

          chaotic-attractors --equation Custom3 \
              --a -2.17 --b -2.7 --c -2.08 --d -2.83 \
              --x-start 0 --y-start 0 --format all --seed 42

          chaotic-attractors --mode search --equation Custom3 \
              --range-min -3 --range-max 3 --num-to-find 10 \
              --max-attempts 50000 --seed 42
        """)
    parser = argparse.ArgumentParser(
        description="Search, validate, and visualize chaotic attractors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument("--mode", choices=["generate", "search"], default="generate")
    parser.add_argument(
        "--equation",
        choices=EQUATION_LIBRARY,
        default=DEFAULT_EQUATION,
        help=f"equation system (default: {DEFAULT_EQUATION})",
    )
    for name in ("a", "b", "c", "d"):
        parser.add_argument(f"--{name}", type=float, metavar="FLOAT")

    parser.add_argument(
        "--test-iter", type=int, default=DEFAULT_TEST_ITERATIONS, metavar="INT"
    )
    parser.add_argument(
        "--final-iter", type=int, default=DEFAULT_FINAL_ITERATIONS, metavar="INT"
    )
    parser.add_argument("--burn-in", type=int, default=DEFAULT_BURN_IN, metavar="INT")
    parser.add_argument(
        "--lyapunov-iter",
        type=int,
        default=DEFAULT_LYAPUNOV_ITERATIONS,
        metavar="INT",
    )
    parser.add_argument(
        "--kde-sample-size",
        type=int,
        default=DEFAULT_KDE_SAMPLE_SIZE,
        metavar="INT",
    )
    parser.add_argument("--x-start", type=float, metavar="FLOAT")
    parser.add_argument("--y-start", type=float, metavar="FLOAT")
    parser.add_argument("--format", choices=["all", "pdf", "png", "svg"], default="png")
    parser.add_argument("--info-panel", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, metavar="PATH")
    parser.add_argument("--seed", type=int, metavar="INT")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace output files with the same names",
    )
    parser.add_argument(
        "--vector-points",
        action="store_true",
        help="keep every point as a vector element in PDF/SVG output",
    )

    parser.add_argument(
        "--range-min", type=float, default=DEFAULT_RANGE_MIN, metavar="FLOAT"
    )
    parser.add_argument(
        "--range-max", type=float, default=DEFAULT_RANGE_MAX, metavar="FLOAT"
    )
    parser.add_argument("--decimals", type=int, default=DEFAULT_DECIMALS, metavar="INT")
    parser.add_argument(
        "--num-to-find", type=int, default=DEFAULT_NUMBER, metavar="INT"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS, metavar="INT"
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
        metavar="INT",
    )
    return parser


def _resolve_starting_values(
    args: argparse.Namespace, *, search_mode: bool
) -> tuple[float, float]:
    defaults = (
        DEFAULT_SEARCH_START
        if search_mode
        else DEFAULT_STARTS.get(args.equation, DEFAULT_SEARCH_START)
    )
    x_start = defaults[0] if args.x_start is None else args.x_start
    y_start = defaults[1] if args.y_start is None else args.y_start
    return x_start, y_start


def _generate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    supplied = {name: getattr(args, name) for name in ("a", "b", "c", "d")}
    if args.equation == DEFAULT_EQUATION:
        params = {
            name: DEFAULT_PARAMS[name] if value is None else value
            for name, value in supplied.items()
        }
    else:
        missing = [name for name, value in supplied.items() if value is None]
        if missing:
            flags = ", ".join(f"--{name}" for name in missing)
            parser.error(f"generate mode requires {flags} for equation {args.equation}")
        params = {name: float(value) for name, value in supplied.items()}

    x_start, y_start = _resolve_starting_values(args, search_mode=False)
    data = prepare_generate_data(
        params=params,
        equation_id=args.equation,
        test_iterations=args.test_iter,
        final_iterations=args.final_iter,
        x_start=x_start,
        y_start=y_start,
        burn_in=args.burn_in,
        lyapunov_iterations=args.lyapunov_iter,
        kde_sample_size=args.kde_sample_size,
        seed=args.seed,
    )
    print(f"Finite-time Lyapunov exponent: {data['lyapunov_exponent']:.4f}")
    print(f"Random seed: {data['seed']}")
    save_attractor(
        data=data,
        x_start=x_start,
        y_start=y_start,
        save_format=args.format,
        output_dir=args.output_dir,
        include_info=args.info_panel,
        overwrite=args.overwrite,
        rasterized_points=not args.vector_points,
    )


def _search(args: argparse.Namespace) -> None:
    x_start, y_start = _resolve_starting_values(args, search_mode=True)
    ranges = dict.fromkeys(("a", "b", "c", "d"), (args.range_min, args.range_max))
    search_attractors(
        equation_id=args.equation,
        num_to_find=args.num_to_find,
        max_attempts=args.max_attempts,
        parameter_ranges=ranges,
        test_iterations=args.test_iter,
        final_iterations=args.final_iter,
        x_start=x_start,
        y_start=y_start,
        decimals=args.decimals,
        include_info=args.info_panel,
        save_format=args.format,
        output_dir=args.output_dir,
        progress_interval=args.progress_interval,
        burn_in=args.burn_in,
        lyapunov_iterations=args.lyapunov_iter,
        kde_sample_size=args.kde_sample_size,
        seed=args.seed,
        overwrite=args.overwrite,
        rasterized_points=not args.vector_points,
    )


def main() -> None:
    """Run the requested CLI mode."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.mode == "generate":
            _generate(args, parser)
        else:
            _search(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except (FileExistsError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
