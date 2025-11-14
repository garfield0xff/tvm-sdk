#ifndef TVM_PYTHON_BRIDGE_H
#define TVM_PYTHON_BRIDGE_H

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <string>

namespace py = pybind11;

namespace tvm_sdk {
namespace bridge {

/**
 * @brief Python Bridge for TVM integration
 *
 * This class provides utilities to call Python code from C++,
 * specifically for importing TVM and getting version information.
 */
class PythonBridge {
public:
    /**
     * @brief Initialize Python interpreter if not already initialized
     */
    static void initialize();

    /**
     * @brief Finalize Python interpreter
     */
    static void finalize();

    /**
     * @brief Import TVM and get its version
     * @return TVM version string (e.g., "0.22.0")
     * @throws std::runtime_error if TVM is not installed or import fails
     */
    static std::string get_tvm_version();

    /**
     * @brief Check if TVM is available
     * @return true if TVM can be imported, false otherwise
     */
    static bool is_tvm_available();

    /**
     * @brief Import a Python module
     * @param module_name Name of the module to import
     * @return Python module object
     * @throws std::runtime_error if import fails
     */
    static py::object import_module(const std::string& module_name);

private:
    static bool is_initialized_;
};

} // namespace bridge
} // namespace tvm_sdk

#endif // TVM_PYTHON_BRIDGE_H
