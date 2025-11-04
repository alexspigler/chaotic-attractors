# Chaotic-Attractors

A Python application for generating and visualizing 4-parameter chaotic attractors using iterative nonlinear dynamical systems.

## Project Overview

This program generates publication-quality visualizations of chaotic attractors by computing millions of iterations from nonlinear recursive equations. Despite deterministic rules, these systems produce intricate, unpredictable patterns.

**Key Features:**
- 5 classic attractor equations (Clifford, Tinkerbell, Fractal Dreams, Peter de Jong, and Johnny Svensson)
- 4 custom-discovered dynamical systems
- Density-based coloring using Gaussian KDE
- Multiple export formats (PNG, PDF, SVG)
- Automatic equation labeling with proper mathematical formatting
- Configurable parameters for exploration

## Gallery

<p align="center">
  <img src="images/Tinkerbell.png" width="270"/>
  <img src="images/Custom2.png" width="270"/>
  <img src="images/Custom3.png" width="270"/>
</p>

**Top Left:** Tinkerbell, &nbsp; $a=0.9$, &nbsp; $b=-0.6013$, &nbsp; $c=2$ &nbsp;, $d=0.5$, &nbsp; $x_0=-0.72$, &nbsp; $y_0=-0.64$  
**Top Right:** Custom2, &nbsp; $a = 0.73$, &nbsp; $b = - 2.6$, &nbsp; $c = 2.31$, &nbsp; $d = 1.65$, &nbsp; $x_0=0$, &nbsp; $y_0=0$  
**Bottom:** Custom3, &nbsp; $a= -2.17$, &nbsp; $b = -2.7$, &nbsp; $c = - 2.08$, &nbsp; $d = -2.83$, &nbsp; $x_0=0$, &nbsp; $y_0=0$  

## Methodology

The program uses a three-stage pipeline:

1. **Iteration** - Generate a default 2M coordinate pairs using recursive equations of the form:
   - $x_{i+1}=f(x_i,y_i,a,b,c,d)$
   - $y_{i+1}=g(x_i,y_i,a,b,c,d)$

2. **Density Calculation** - Apply Gaussian kernel density estimation (KDE) to reveal structure in the point cloud

3. **Visualization** - Map density values to custom color gradients with configurable transparency and point size

## Mathematical Background

Chaotic attractors are bounded regions in phase space where deterministic systems exhibit sensitive dependence on initial conditions. Despite their chaotic behavior, these systems produce intricate geometric structures with fractal-like properties.

## Technologies

**Language:** Python

**Core Libraries:**
- `NumPy` - Fast array operations for multi-million point trajectories
- `SciPy` - Gaussian KDE for density estimation
- `Matplotlib` - Publication-quality plotting with custom colormaps

## Quick Start

### Prerequisites
```bash
pip install numpy scipy matplotlib
```

### Default Run
```bash
python chaotic_attractors.py
```
The default run will plot the famous Tinkerbell attractor using 2M iterations and the parameters:  
$$a=0.9 \text{,}\quad b=-0.6013 \text{,}\quad c=2 \text{,}\quad d=0.5$$  
and the starting position:  
$$x_0=-0.72 \text{, } y_0=-0.64$$  
Output files are saved in the current directory as PNG, PDF, and SVG formats.

### Basic Usage
To explore different attractors, open `chaotic_attractors.py` and edit the appropriate lines:

1. **Select an attractor equation:**
```python
ACTIVE_EQUATION = "Custom3"
```

2. **Set parameters:**
```python
params = {
    'a': -2.17,
    'b': -2.7,
    'c': -2.08,
    'd': -2.83
}
```

3. **Run:**
```bash
python chaotic_attractors.py
```

## Configuration

### Iteration Count
Higher iterations produce smoother results but require more memory:
```python
ITERATIONS_DEFAULT = 2_000_000
```

### Starting Position
Some attractors (e.g., Tinkerbell) require non-zero starting points:
```python
X_START = 0.5
Y_START = 0.5
```

### Color Schemes
Choose from built-in palettes or custom gradients:
```python
COLOR_METHOD = "gradient"  # Options: "viridis", "gradient", "gradient3", "gradientn"

GRADIENT_LOW = "lightblue"
GRADIENT_HIGH = "darkviolet"
```

## Available Attractors

### Classic Systems
- **Clifford**
- **Tinkerbell**
- **Fractal Dreams**
- **Peter de Jong**
- **Johnny Svensson** 

### Custom Discoveries
- **Custom1, Custom2, Custom3, Custom4** - Novel attractors discovered through parameter exploration
- Each produces visually distinct patterns and structural complexity

## Output Files

The program generates two versions of each visualization:

1. **Standard** (`attractor_output.*`) - Clean visualization without annotations
2. **Annotated** (`attractor_with_eq.*`) - Includes equations and parameters

All files are saved in PNG (for sharing), PDF (for documents), and SVG (for scaling).

## Implementation Details

**Performance:**
- Efficient computation using NumPy's optimized mathematical functions
- KDE uses sampling to manage memory usage
- Density normalization is used for consistent coloring

**Numerical Stability:**
- Automatic handling of NaN and Inf values through boolean masking
- Validation check requiring a minimum of 10k valid points
- Graceful degradation to uniform coloring if KDE fails

## Technical Notes

- The program uses `eval()` for runtime equation selection from the equation library
- Point generation is sequential (not parallelized) due to recursive dependencies
- Color gradients use matplotlib's `LinearSegmentedColormap` for smooth transitions
- Automatic equation labeling uses matplotlib's math text parser

---