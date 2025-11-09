"""
Equation definitions for 4-parameter chaotic attractors.
"""

EQUATION_LIBRARY = {
    # Clifford
    "Clifford": {
        "x_eq": "np.sin(a * y[n-1]) + c * np.cos(a * x[n-1])",
        "y_eq": "np.sin(b * x[n-1]) + d * np.cos(b * y[n-1])",
    },
    # Tinkerbell (x_0, y_0 != 0, 0)
    "Tinkerbell": {
        "x_eq": "x[n-1]**2 - y[n-1]**2 + a * x[n-1] + b * y[n-1]",
        "y_eq": "2 * x[n-1] * y[n-1] + c * x[n-1] + d * y[n-1]",
    },
    # Fractal Dreams
    "Fractal_Dreams": {
        "x_eq": "np.sin(b * y[n-1]) + c * np.sin(b * x[n-1])",
        "y_eq": "np.sin(a * x[n-1]) + d * np.sin(a * y[n-1])",
    },
    # Peter de Jong
    "Peter_Jong": {
        "x_eq": "np.sin(a * y[n-1]) - np.cos(b * x[n-1])",
        "y_eq": "np.sin(c * x[n-1]) - np.cos(d * y[n-1])",
    },
    # Johnny Svensson
    "Johnny_Svensson": {
        "x_eq": "d * np.sin(a * x[n-1]) - np.sin(b * y[n-1])",
        "y_eq": "c * np.cos(a * x[n-1]) + np.cos(b * y[n-1])",
    },
    # Custom 1
    "Custom1": {
        "x_eq": "np.sin(np.cos(a * y[n-1])) + c * np.cos(a * x[n-1])",
        "y_eq": "np.sin(np.cos(b * x[n-1])) + d * np.cos(b * y[n-1])",
    },
    # Custom 2
    "Custom2": {
        "x_eq": "a * (np.exp(np.cos(x[n-1])) - np.pi / 2) + b * (np.exp(np.sin(y[n-1])) - np.pi / 2)",
        "y_eq": "c * (np.exp(np.sin(x[n-1])) - np.pi / 2) + d * (np.exp(np.cos(y[n-1])) - np.pi / 2)",
    },
    # Custom 3
    "Custom3": {
        "x_eq": "a * np.exp(np.arcsinh(x[n-1])) - b * np.exp(np.sin(y[n-1]))",
        "y_eq": "c * np.exp(np.arcsinh(y[n-1])) - d * np.exp(np.sin(x[n-1]))",
    },
    # Custom 4 (x_0, y_0 != 0, 0)
    "Custom4": {
        "x_eq": "x[n-1]**2 - y[n-1]**2 + a * np.sin(x[n-1]) + b * np.sin(b * y[n-1])",
        "y_eq": "a * x[n-1] * y[n-1] + c * x[n-1] + d * np.sin(y[n-1])",
    },
}
