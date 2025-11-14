#include "bridge/python_bridge.h"
#include <iostream>
#include <stdexcept>

namespace tvm_sdk {
namespace bridge {

bool PythonBridge::is_initialized_ = false;

void PythonBridge::initialize() {
    if (!is_initialized_) {
        // Initialize Python interpreter
        // Note: py::scoped_interpreter is RAII-based, so we don't use it here
        // Instead, we manually initialize if needed
        if (!Py_IsInitialized()) {
            py::initialize_interpreter();
        }
        is_initialized_ = true;
    }
}

void PythonBridge::finalize() {
    if (is_initialized_) {
        // Only finalize if we initialized it
        if (Py_IsInitialized()) {
            py::finalize_interpreter();
        }
        is_initialized_ = false;
    }
}

std::string PythonBridge::get_tvm_version() {
    // [1] Ensure Python is initialized
    initialize();

    try {
        // [2] GIL is automatically acquired by pybind11 functions
        py::gil_scoped_acquire gil;

        // [3] Import TVM module
        py::object tvm = py::module_::import("tvm");

        // [4] Get version attribute
        // TVM version is in tvm.__version__
        py::object version_obj = tvm.attr("__version__");

        // [5] Convert to C++ string
        std::string version = py::cast<std::string>(version_obj);

        return version;

    } catch (const py::error_already_set& e) {
        // [6] Error handling - Python exception occurred
        throw std::runtime_error(
            std::string("Failed to get TVM version: ") + e.what()
        );
    }
}

bool PythonBridge::is_tvm_available() {
    initialize();

    try {
        py::gil_scoped_acquire gil;
        py::module_::import("tvm");
        return true;
    } catch (const py::error_already_set&) {
        return false;
    }
}

py::object PythonBridge::import_module(const std::string& module_name) {
    // [1] Ensure Python is initialized
    initialize();

    try {
        // [2] GIL acquisition
        py::gil_scoped_acquire gil;

        // [3] Import the module
        py::object module = py::module_::import(module_name.c_str());

        return module;

    } catch (const py::error_already_set& e) {
        // [4] Error handling
        throw std::runtime_error(
            std::string("Failed to import module '") + module_name + "': " + e.what()
        );
    }
}

} // namespace bridge
} // namespace tvm_sdk
