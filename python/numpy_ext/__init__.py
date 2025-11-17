"""
NumPy Extensions for TVM SDK

This module provides NumPy utility functions for C++ integration.
"""

from .numpy_samples import (
    add_arrays,
    matrix_multiply,
    create_random_array,
    array_statistics,
    reshape_array,
    dot_product
)

__all__ = [
    'add_arrays',
    'matrix_multiply',
    'create_random_array',
    'array_statistics',
    'reshape_array',
    'dot_product'
]
