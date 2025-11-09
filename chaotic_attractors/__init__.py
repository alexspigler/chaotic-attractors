"""
chaotic_attractors package

Provides tools to search for, generate, evaluate, and visualize 4-parameter chaotic attractors.
"""

from .equations import EQUATION_LIBRARY

from .core import (
    ALPHA_DEFAULT,
    POINT_SIZE_DEFAULT,
    COLOR_METHOD,
    VIRIDIS_PALETTE,
    GRADIENT_LOW,
    GRADIENT_HIGH,
    GRADIENT3_LOW,
    GRADIENT3_MID,
    GRADIENT3_HIGH,
    GRADIENT3_MIDPOINT,
    GRADIENTN_COLORS,
    GRADIENTN_VALUES,
    _compile_equation,
    _get_equation_functions,
    generate_chaotic,
    prepare_generate_data,
    create_colormap,
    plot_chaotic,
    save_attractor,
    convert_to_math_text,
    create_attractor_with_eq,
)

from .search import (
    generate_random,
    evaluate_attractor,
    search_attractors,
    prepare_search_data
)
