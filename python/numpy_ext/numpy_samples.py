"""
NumPy Sample Functions for C++ Integration
"""

import numpy as np


def add_arrays(a, b):
    """
    Add two numpy arrays element-wise.

    Args:
        a: numpy array or list
        b: numpy array or list

    Returns:
        numpy array: element-wise sum
    """
    return np.add(a, b)


def matrix_multiply(a, b):
    """
    Multiply two matrices.

    Args:
        a: 2D numpy array or list
        b: 2D numpy array or list

    Returns:
        numpy array: matrix product
    """
    return np.matmul(a, b)


def create_random_array(shape, seed=42):
    """
    Create a random numpy array with given shape.

    Args:
        shape: tuple of dimensions (e.g., (3, 4))
        seed: random seed for reproducibility

    Returns:
        numpy array: random array
    """
    np.random.seed(seed)
    return np.random.rand(*shape)


def array_statistics(arr):
    """
    Compute statistics of a numpy array.

    Args:
        arr: numpy array

    Returns:
        dict: statistics including mean, std, min, max
    """
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'shape': arr.shape,
        'dtype': str(arr.dtype)
    }


def reshape_array(arr, new_shape):
    """
    Reshape a numpy array to new dimensions.

    Args:
        arr: numpy array
        new_shape: tuple of new dimensions

    Returns:
        numpy array: reshaped array
    """
    return np.reshape(arr, new_shape)


def dot_product(a, b):
    """
    Compute dot product of two vectors.

    Args:
        a: 1D numpy array or list
        b: 1D numpy array or list

    Returns:
        float: dot product result
    """
    return float(np.dot(a, b))


if __name__ == "__main__":
    # Test functions
    print("=== NumPy Sample Functions Test ===\n")

    # Test add_arrays
    a = [1, 2, 3]
    b = [4, 5, 6]
    result = add_arrays(a, b)
    print(f"add_arrays({a}, {b}) = {result}")

    # Test matrix_multiply
    mat_a = [[1, 2], [3, 4]]
    mat_b = [[5, 6], [7, 8]]
    result = matrix_multiply(mat_a, mat_b)
    print(f"\nmatrix_multiply:\n{result}")

    # Test create_random_array
    random_arr = create_random_array((2, 3))
    print(f"\ncreate_random_array((2, 3)):\n{random_arr}")

    # Test array_statistics
    stats = array_statistics(random_arr)
    print(f"\narray_statistics:\n{stats}")

    # Test dot_product
    vec_a = [1, 2, 3]
    vec_b = [4, 5, 6]
    dot = dot_product(vec_a, vec_b)
    print(f"\ndot_product({vec_a}, {vec_b}) = {dot}")
