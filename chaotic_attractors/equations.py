"""
Equation definitions for 4-parameter chaotic attractors.
"""

EQUATION_LIBRARY = {
    # Clifford
    "Clifford": {
        "x_eq": "np.sin(a * y[i-1]) + c * np.cos(a * x[i-1])",
        "y_eq": "np.sin(b * x[i-1]) + d * np.cos(b * y[i-1])"
    },
    
    # Tinkerbell (x_0, y_0 != 0, 0)
    "Tinkerbell": {
        "x_eq": "x[i-1]**2 - y[i-1]**2 + a * x[i-1] + b * y[i-1]",
        "y_eq": "2 * x[i-1] * y[i-1] + c * x[i-1] + d * y[i-1]"
    },
    
    # Fractal Dreams
    "Fractal_Dreams": {
        "x_eq": "np.sin(b * y[i-1]) + c * np.sin(b * x[i-1])",
        "y_eq": "np.sin(a * x[i-1]) + d * np.sin(a * y[i-1])"
    },
    
    # Peter de Jong
    "Peter_Jong": {
        "x_eq": "np.sin(a * y[i-1]) - np.cos(b * x[i-1])",
        "y_eq": "np.sin(c * x[i-1]) - np.cos(d * y[i-1])"
    },
    
    # Johnny Svensson
    "Johnny_Svensson": {
        "x_eq": "d * np.sin(a * x[i-1]) - np.sin(b * y[i-1])",
        "y_eq": "c * np.cos(a * x[i-1]) + np.cos(b * y[i-1])"
    },
    
    # Custom 1
    "Custom1": {
        "x_eq": "np.sin(np.cos(a * y[i-1])) + c * np.cos(a * x[i-1])",
        "y_eq": "np.sin(np.cos(b * x[i-1])) + d * np.cos(b * y[i-1])"
    },
    
    # Custom 2
    "Custom2": {
        "x_eq": "a * (np.exp(np.cos(x[i-1])) - np.pi / 2) + b * (np.exp(np.sin(y[i-1])) - np.pi / 2)",
        "y_eq": "c * (np.exp(np.sin(x[i-1])) - np.pi / 2) + d * (np.exp(np.cos(y[i-1])) - np.pi / 2)"
    },
    
    # Custom 3
    "Custom3": {
        "x_eq": "a * np.exp(np.arcsinh(x[i-1])) - b * np.exp(np.sin(y[i-1]))",
        "y_eq": "c * np.exp(np.arcsinh(y[i-1])) - d * np.exp(np.sin(x[i-1]))"
    },
    
    # Custom 4 (x_0, y_0 != 0, 0)
    "Custom4": {
        "x_eq": "x[i-1]**2 - y[i-1]**2 + a * np.sin(x[i-1]) + b * np.sin(b * y[i-1])",
        "y_eq": "a * x[i-1] * y[i-1] + c * x[i-1] + d * np.sin(y[i-1])"
    }
}
