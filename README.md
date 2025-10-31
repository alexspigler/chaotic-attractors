# Chaotic-Attractors
Generate and visualize chaotic attractors with specific parameters.

## Purpose

Use this program when you know the parameters you want to visualize. You can:
- Recreating famous attractors (Clifford, Tinkerbell, etc. -- with their specific known parameters)
- Test new parameters with known attractors
- Create high-quality visualizations
- Experiment with different color schemes and gradients

## Quick Start

```bash
python \path\to\attractor_runner.py
```

This runs with default settings (Clifford attractor, with $a=2.1$, $b=-0.8$, $c=-0.2$, $d=2.5$) and creates output files in the current directory.

## Basic Usage

### 1. Choose Your Equation

Edit the `ACTIVE_EQUATION` variable at the bottom of the file:

```python
ACTIVE_EQUATION = "Clifford"  # or any equation from the library
```

**Available equations:**
- `Clifford`
- `Tinkerbell`
- `Fractal_Dreams`
- `Peter_Jong`
- `Johnny_Svensson`
- `Custom1` through `Custom11` - Custom discovered attractors

### 2. Set Parameters

```python
params = {
    'a': 2.1,
    'b': -0.8,
    'c': -0.2,
    'd': 2.5
}
```

Each equation uses 4 parameters (a, b, c, d). Different values create completely different patterns.

### 3. Run Program

```bash
python attractor_runner.py
```
<br>

**The program will by default:**
1. Generate the attractor (with 2 million iterations)
2. Calculate density via `gaussian_kde` in the `scipy.stats` library
3. Use that density to create a colormap gradient (with light blue for the low density areas and with purple for the high density areas)
3. Save in all formats (PNG, PDF, SVG)
4. Create versions with and without equation and parameter info at the bottom

<br>

## Configuration Options

### Iterations

Control how many points to generate (more = smoother, slower, more memory intensive, larger file sizes):

```python
ITERATIONS_DEFAULT = 1_000_000  # Default
# ITERATIONS_DEFAULT = 100_000   # Fast preview (see if the attractor is valid or if it collapses or diverges)
# ITERATIONS_DEFAULT = 5_000_000 # Ultra high quality, very memory intensive
```

### Starting Point

Some attractors are sensitive to starting position, such as `Tinkerbell` and `Custom11`.
With these you will be hard pressed to find a valid attractor with `X_START, Y_START = 0.0`.  I suggest changing both to 0.5 and seing if you can find attractors that way.

```python
X_START = 0.5
Y_START = 0.5
```

### Visualization Settings

If your attractor is very spread out, you may want to try increasing the `ALPHA_DEFAULT` as well as the `POINT_SIZE_DEFAULT` 

```python
# Point appearance
ALPHA_DEFAULT = 0.3        # Transparency (0-1, 0=completely transparent)
POINT_SIZE_DEFAULT = 0.15  # Size of each point
```

### Color Schemes

First select the coloring option you'd like.
* gradient = 2 color gradient
```python
# Color method
COLOR_METHOD = "gradient"  # Options: "viridis", "gradient", "gradient3", "gradientn"
```

