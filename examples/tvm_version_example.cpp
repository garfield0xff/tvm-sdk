/**
 * @file tvm_version_example.cpp
 * @brief Example of using PythonBridge to get TVM version
 *
 * This example demonstrates how to use the PythonBridge class
 * to import TVM and retrieve its version from C++.
 *
 * Based on the patterns from CPP_TO_PYTHON_CALLS.md:
 * 1. GIL acquisition (handled by pybind11)
 * 2. Import Python module
 * 3. Access module attributes
 * 4. Convert Python objects to C++ types
 * 5. Error handling
 */

#include "bridge/python_bridge.h"
#include <iostream>
#include <exception>

using namespace tvm_sdk::bridge;

int main() {
    try {
        // Initialize Python interpreter
        std::cout << "Initializing Python interpreter..." << std::endl;
        PythonBridge::initialize();

        // Check if TVM is available
        std::cout << "Checking if TVM is available..." << std::endl;
        if (PythonBridge::is_tvm_available()) {
            std::cout << "✓ TVM is available" << std::endl;

            // Get TVM version
            std::string version = PythonBridge::get_tvm_version();
            std::cout << "TVM Version: " << version << std::endl;

        } else {
            std::cout << "✗ TVM is not available" << std::endl;
            std::cout << "Please install TVM: pip install apache-tvm" << std::endl;
        }

        // Example: Import other Python modules
        try {
            std::cout << "\nImporting numpy..." << std::endl;
            auto numpy = PythonBridge::import_module("numpy");
            auto np_version = numpy.attr("__version__");
            std::cout << "NumPy Version: " << py::cast<std::string>(np_version) << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Could not import numpy: " << e.what() << std::endl;
        }

        // Cleanup
        std::cout << "\nFinalizing Python interpreter..." << std::endl;
        PythonBridge::finalize();

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
