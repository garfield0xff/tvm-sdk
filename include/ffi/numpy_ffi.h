#ifndef TVM_NUMPY_FFI_H
#define TVM_NUMPY_FFI_H

#include "python_hook.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <map>

namespace tvm_sdk {
namespace ffi {

/**
 * @brief NumPy FFI (Foreign Function Interface)
 *
 * Provides high-level interface to call NumPy functions from C++
 * using the python/numpy_ext/numpy_samples.py module.
 */
class NumPyFFI {
public:
    /**
     * @brief Add two arrays element-wise
     * @param a First array
     * @param b Second array
     * @return Result array
     */
    static py::array add_arrays(const py::array& a, const py::array& b);

    /**
     * @brief Multiply two matrices
     * @param a First matrix
     * @param b Second matrix
     * @return Result matrix
     */
    static py::array matrix_multiply(const py::array& a, const py::array& b);

    /**
     * @brief Create random array with given shape
     * @param shape Shape as vector (e.g., {3, 4})
     * @param seed Random seed
     * @return Random array
     */
    static py::array create_random_array(const std::vector<int>& shape, int seed = 42);

    /**
     * @brief Compute array statistics
     * @param arr Input array
     * @return Statistics as map (mean, std, min, max, etc.)
     */
    static std::map<std::string, double> array_statistics(const py::array& arr);

    /**
     * @brief Reshape array to new dimensions
     * @param arr Input array
     * @param new_shape New shape as vector
     * @return Reshaped array
     */
    static py::array reshape_array(const py::array& arr, const std::vector<int>& new_shape);

    /**
     * @brief Compute dot product of two vectors
     * @param a First vector
     * @param b Second vector
     * @return Dot product result
     */
    static double dot_product(const py::array& a, const py::array& b);

    // Conversion utilities

    /**
     * @brief Convert std::vector to numpy array
     * @tparam T Element type
     * @param vec std::vector
     * @return NumPy array
     */
    template<typename T>
    static py::array_t<T> vector_to_numpy(const std::vector<T>& vec) {
        // Ensure Python is initialized before creating numpy arrays
        if (!PythonHook::is_initialized()) {
            PythonHook::initialize();
        }
        return py::array_t<T>(vec.size(), vec.data());
    }

    /**
     * @brief Convert 2D std::vector to numpy array
     * @tparam T Element type
     * @param vec 2D std::vector
     * @return NumPy array
     */
    template<typename T>
    static py::array_t<T> vector2d_to_numpy(const std::vector<std::vector<T>>& vec) {
        // Ensure Python is initialized before creating numpy arrays
        if (!PythonHook::is_initialized()) {
            PythonHook::initialize();
        }

        if (vec.empty()) {
            return py::array_t<T>();
        }

        size_t rows = vec.size();
        size_t cols = vec[0].size();

        py::array_t<T> result({rows, cols});
        auto buf = result.request();
        T* ptr = static_cast<T*>(buf.ptr);

        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                ptr[i * cols + j] = vec[i][j];
            }
        }

        return result;
    }

    /**
     * @brief Convert numpy array to std::vector
     * @tparam T Element type
     * @param arr NumPy array
     * @return std::vector
     */
    template<typename T>
    static std::vector<T> numpy_to_vector(const py::array_t<T>& arr) {
        // Ensure Python is initialized before accessing numpy arrays
        if (!PythonHook::is_initialized()) {
            PythonHook::initialize();
        }
        auto buf = arr.request();
        T* ptr = static_cast<T*>(buf.ptr);
        return std::vector<T>(ptr, ptr + buf.size);
    }

private:
    static constexpr const char* MODULE_NAME = "numpy_ext.numpy_samples";
};

} // namespace ffi
} // namespace tvm_sdk

#endif // TVM_NUMPY_FFI_H
