#include "ffi/numpy_ffi.h"
#include "python_hook.h"
#include <stdexcept>

namespace tvm_sdk {
namespace ffi {d

py::array NumPyFFI::add_arrays(const py::array& a, const py::array& b) {
    py::object result = PythonHook::call_function(MODULE_NAME, "add_arrays", a, b);
    return py::cast<py::array>(result);
}

py::array NumPyFFI::matrix_multiply(const py::array& a, const py::array& b) {
    py::object result = PythonHook::call_function(MODULE_NAME, "matrix_multiply", a, b);
    return py::cast<py::array>(result);
}

py::array NumPyFFI::create_random_array(const std::vector<int>& shape, int seed) {
    py::tuple py_shape = py::cast(shape);
    py::object result = PythonHook::call_function(MODULE_NAME, "create_random_array", py_shape, seed);
    return py::cast<py::array>(result);
}

std::map<std::string, double> NumPyFFI::array_statistics(const py::array& arr) {
    py::object result = PythonHook::call_function(MODULE_NAME, "array_statistics", arr);
    py::dict stats = py::cast<py::dict>(result);

    std::map<std::string, double> cpp_stats;
    cpp_stats["mean"] = py::cast<double>(stats["mean"]);
    cpp_stats["std"] = py::cast<double>(stats["std"]);
    cpp_stats["min"] = py::cast<double>(stats["min"]);
    cpp_stats["max"] = py::cast<double>(stats["max"]);

    return cpp_stats;
}

py::array NumPyFFI::reshape_array(const py::array& arr, const std::vector<int>& new_shape) {
    py::tuple py_shape = py::cast(new_shape);
    py::object result = PythonHook::call_function(MODULE_NAME, "reshape_array", arr, py_shape);
    return py::cast<py::array>(result);
}

double NumPyFFI::dot_product(const py::array& a, const py::array& b) {
    py::object result = PythonHook::call_function(MODULE_NAME, "dot_product", a, b);
    return py::cast<double>(result);
}

} // namespace ffi
} // namespace tvm_sdk
