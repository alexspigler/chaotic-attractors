# Chaotic Attractors: Computational Exploration & Visualization

A Python framework for discovering and visualizing 4-parameter chaotic dynamical systems through algorithmic parameter space exploration and statistical quality filtering.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Gallery

<p align="center">
  <img src="images/Tinkerbell.png" width="270"/>
  <img src="images/Custom2.png" width="270"/>
  <img src="images/Custom3.png" width="270"/>
</p>


**From left to right:** Tinkerbell, Custom2, and Custom3
| Equation | Parameters | Initial Position |
|-----------|-------------|---------------|
| **Tinkerbell** | $a=0.9$, $b=-0.6013$, $c=2.0$, $d=0.5$ | $x_0=-0.72$, $y_0=-0.64$ |
| **Custom2**    | $a=0.73$, $b=-2.6$, $c=2.31$, $d=1.65$ | $x_0=0$, $y_0=0$ |
| **Custom3**    | $a=-2.17$, $b=-2.7$, $c=-2.08$, $d=-2.83$ | $x_0=0$, $y_0=0$ |

---

## Overview

This project implements a computational pipeline for generating, filtering, and visualizing chaotic attractors---complex fractal structures that emerge from deterministic iterative systems. The toolkit features:

- **Automated Discovery**: Stochastic parameter search across 4-dimensional space
- **Statistical Filtering**: Multi-criteria quality assessment using geometric properties and uniqueness metrics
- **High-Performance Computation**: Vectorized NumPy operations generating 2M+ trajectory points
- **Density Estimation**: Gaussian kernel density estimation (KDE) for structure-revealing visualization
- **Production Pipeline**: Comprehensive testing (50+ unit tests), modular architecture, and publication-ready exports

**Technical Highlights:**
- Efficient iterative map evaluation with runtime equation compilation
- Multi-stage filtering (divergence, collapse, aspect ratio, uniqueness) to reject uninteresting/unviable candidates
- Custom scoring function balancing geometric and statistical criteria
- Publication-quality output in multiple formats (PNG, PDF, SVG)

---

## Mathematical Background

Chaotic attractors are generated through discrete-time dynamical systems:

$$x_{n+1} = f(x_n, y_n; a, b, c, d)$$
$$y_{n+1} = g(x_n, y_n; a, b, c, d)$$

where $(x_0, y_0)$ are initial conditions and $(a, b, c, d)$ are system parameters.

### Example Systems

**Tinkerbell:**
- $x_{n+1} = x_n^2 - y_n^2 + ax_n + by_n$
- $y_{n+1} = 2x_ny_n + cx_n + dy_n$

**Clifford:**
- $x_{n+1} = \sin(ay_n) + c\cos(ax_n)$
- $y_{n+1} = \sin(bx_n) + d\cos(by_n)$

The library includes 9 equation systems: 5 classical attractors (Clifford, Tinkerbell, Fractal Dreams, Peter de Jong, Johnny Svensson) and 4 custom-designed variants.

---

## Key Features

### 1. Automated Parameter Discovery
Random search algorithm with configurable bounds and precision:
```python
# Search for 15 high-quality attractors in specified parameter ranges
python -m chaotic_attractors.__main__ --mode search \
    --equation Clifford \
    --range-min -3.0 --range-max 3.0 \
    --decimals 2
```

**Algorithm workflow:**
1. Sample parameters uniformly from discrete grid (e.g., 0.01 precision -- roughly 13 billion possible combinations)
2. Generate 10K test points for rapid evaluation
3. Apply multi-stage filtering:
   - **Divergence check**: Reject if range > 500 units
   - **Collapse check**: Reject if range < 0.25 units  
   - **Periodicity filter**: Reject if unique point ratio < 0.25
   - **Aspect ratio**: Reject if dimensions differ by >4×
4. Regenerate accepted candidates at 2M points with a final check on full attractor plots
5. Plot with custom color maps
5. Export as PDFs with equation annotations + CSV parameter summary

### 2. Quality Scoring System

Attractors are ranked using a Mean Squared Error (MSE) metric that measures deviation from ideal characteristics:

$$\text{score} = \left(\frac{r_{\text{aspect}} - r_{\text{ideal,aspect}}}{r_{\text{max}} - 1}\right)^2 + \left(\frac{r_{\text{unique}} - r_{\text{ideal,unique}}}{r_{\text{max,unique}} - r_{\text{min,unique}}}\right)^2$$

**Parameters:**
- $r_{\text{aspect}}$ = aspect ratio (longer dimension / shorter dimension)
- $r_{\text{unique}}$ = fraction of unique points at 2 decimal precision
- $r_{\text{ideal,aspect}} = 1.5$ (target aspect ratio for visual balance)
- $r_{\text{ideal,unique}} = 0.75$ (target uniqueness to avoid periodic/chaotic extremes)
- Acceptable ranges: aspect $\in [1.0, 4.0]$, unique $\in [0.25, 0.95]$

**Interpretation:**
- **Score = 0**: Perfect match to ideal characteristics
- **Score < 0.1**: Excellent quality

This is equivalent to computing Euclidean distance in normalized parameter space, where each criterion is scaled by its acceptable range to ensure fair weighting.

### 3. Density-Based Visualization

Implements Gaussian KDE for structure revelation:
- Sample 50K points from full trajectory for computational efficiency
- Compute probability density for all 2M points
- Map density to custom colormaps (viridis, gradient, multi-stop)
- Alpha blending and point sizing for aesthetic control

---

## Installation & Setup

```bash
# Clone repository
git clone https://github.com/alexspigler/chaotic-attractors.git
cd chaotic-attractors

# Install package with dependencies
pip install -e .

# Or install with development tools (recommended for contributors)
pip install -e ".[dev]"

# Verify installation by running tests
pytest
```

**Requirements:**
- Python 3.8+
- NumPy, SciPy, Matplotlib (installed automatically)
- pytest, pytest-cov (included with `[dev]` installation)

---

## Usage Examples

### Generate Single Attractor
```python
from chaotic_attractors import prepare_attractor_data, save_attractor

# Define parameters
params = {'a': 0.9, 'b': -0.6013, 'c': 2.0, 'd': 0.5}

# Generate 2M points with density calculation
data = prepare_attractor_data(params, equation_id='Tinkerbell')

# Save in multiple formats with equation panel
save_attractor(
    data,
    filename='my_attractor',
    save_format='all',  # PNG, PDF, SVG
    include_info=True,
    equation_id='Tinkerbell'
)
```

### Command-Line Interface
```bash
# Default Tinkerbell parameters
python -m chaotic_attractors.__main__

# Custom parameters
python -m chaotic_attractors.__main__ \
    --a 0.9 --b -0.6 --c 2.0 --d 0.5 \
    --equation Tinkerbell

# Parameter space search
python -m chaotic_attractors.__main__ \
    --mode search \
    --equation Clifford \
    --range-min -2.5 --range-max 2.5 \
    --decimals 2
```
---

## Project Architecture

```
chaotic-attractors/
├── chaotic_attractors/          # Main package
│   ├── __init__.py              # Public API exports
│   ├── __main__.py              # CLI interface with argparse
│   ├── core.py                  # Generation, KDE, visualization
│   ├── equations.py             # System definitions (9 attractors)
│   └── search.py                # Stochastic search & filtering
├── tests/                       # Comprehensive test suite
│   ├── test_core.py             # Core generation logic
│   ├── test_equations.py        # Equation compilation
│   └── test_search.py           # Parameter search & scoring
├── images/                      # Example outputs
├── .gitignore                   # Version control exclusions
├── LICENSE                      # MIT License
├── pytest.ini                   # Test runner configuration
├── pyproject.toml               # Package metadata and build config
└── README.md                    # Project documentation
```

### Module Responsibilities

**`core.py`**: Numerical computation and rendering
- Runtime equation compilation with restricted `eval` namespace
- Vectorized trajectory generation
- NaN/Inf filtering with early termination
- Gaussian KDE on 50K subsamples
- Custom colormap construction
- Multi-format export with Matplotlib equation rendering

**`search.py`**: Parameter space exploration
- Discrete uniform sampling with fixed precision
- Multi-criteria evaluation function
- Automated file management and CSV logging
- Progress tracking with timing statistics

**`equations.py`**: System library
- String-based equation definitions
- Easy addition of new attractors

---

## Testing & Validation

50+ unit tests across 3 modules ensuring code quality and correctness:

* `test_core.py`: Generation, KDE, visualization, Matplotlib conversion
* `test_equations.py`: All 9 systems validated, parametrized tests
* `test_search.py`: Random sampling, evaluation, edge cases

```bash
# Run full test suite with coverage
pytest

# View detailed coverage report
open htmlcov/index.html
```

Test configuration in `pyproject.toml` automatically enables verbose output, coverage measurement, and HTML report generation.

---

## Technical Highlights

**Performance:** Pre-allocated arrays, early divergence detection, vectorized operations, O(n) KDE sampling

**Stability:** NaN/Inf filtering, minimum point thresholds, restricted eval namespace

**Reproducibility:** Fixed random seeds, deterministic sampling, CSV logging

---

**License**: MIT License - see [LICENSE](LICENSE) file for details.