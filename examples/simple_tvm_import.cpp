/**
 * @file simple_tvm_import.cpp
 * @brief Minimal example of importing TVM from C++
 *
 * This is the simplest possible example showing how to:
 * - Import TVM module
 * - Get TVM version
 *
 * Compile with:
 *   g++ -std=c++17 simple_tvm_import.cpp -I../include -I../third_party/pybind \
 *       -I/usr/include/python3.9 -lpython3.9 -o simple_tvm_import
 */

#include "bridge/python_bridge.h"
#include <iostream>

int main() {
    try {
        // Get TVM version (automatically initializes Python if needed)
        std::string version = tvm_sdk::bridge::PythonBridge::get_tvm_version();
        std::cout << "TVM Version: " << version << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
