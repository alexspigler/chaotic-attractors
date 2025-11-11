"""
chaotic_attractors package

Provides tools to search for, generate, evaluate, and visualize 4-parameter chaotic attractors.
"""

from .core import (
    ALPHA_DEFAULT,
    COLOR_METHOD,
    GRADIENT3_HIGH,
    GRADIENT3_LOW,
    GRADIENT3_MID,
    GRADIENT3_MIDPOINT,
    GRADIENT_HIGH,
    GRADIENT_LOW,
    GRADIENTN_COLORS,
    GRADIENTN_VALUES,
    POINT_SIZE_DEFAULT,
    VIRIDIS_PALETTE,
    _compile_equation,
    _get_equation_functions,
    convert_to_math_text,
    create_attractor_with_eq,
    create_colormap,
    evaluate_attractor_first,
    generate_chaotic,
    plot_chaotic,
    prepare_generate_data,
    save_attractor,
)
from .equations import EQUATION_LIBRARY
from .search import (
    evaluate_attractor_second,
    generate_random,
    prepare_search_data,
    search_attractors,
)
