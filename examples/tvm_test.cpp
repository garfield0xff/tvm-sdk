/**
 * @file tvm_test.cpp
 * @brief Test TVM FFI functionality
 *
 * Demonstrates how to use TVMFFI to interact with TVM from C++
 * using the python/tvm/tvm_samples.py module.
 */

#include "ffi/tvm_ffi.h"
#include "python_hook.h"
#include <iostream>
#include <iomanip>

using namespace tvm_sdk::ffi;
using namespace tvm_sdk;

void print_separator() {
    std::cout << "\n========================================\n";
}

int main() {
    try {
        print_separator();
        std::cout << "TVM FFI Test Suite\n";
        print_separator();

        // Initialize Python (uses TVM_SDK_PYTHON_PATH environment variable)
        PythonHook::initialize();

        // Test 1: Get TVM version
        std::cout << "\n[Test 1] TVM Version\n";
        std::cout << "--------------------\n";
        std::string version = TVMFFI::get_tvm_version();
        std::cout << "TVM Version: " << version << "\n";

        // Test 2: Check TVM modules
        std::cout << "\n[Test 2] TVM Modules\n";
        std::cout << "--------------------\n";
        auto modules = TVMFFI::check_tvm_modules();
        for (const auto& pair : modules) {
            std::cout << "  " << std::setw(10) << std::left << pair.first << ": "
                      << (pair.second ? "✓ Available" : "✗ Not available") << "\n";
        }

        // Test 3: Get TVM target info
        std::cout << "\n[Test 3] TVM Target Info\n";
        std::cout << "------------------------\n";
        auto target_info = TVMFFI::get_tvm_target("llvm");
        for (const auto& pair : target_info) {
            std::cout << "  " << pair.first << ": " << pair.second << "\n";
        }

        // Test 4: Create simple IR
        std::cout << "\n[Test 4] Create Simple IR\n";
        std::cout << "-------------------------\n";
        std::string ir_str = TVMFFI::create_simple_ir();
        std::cout << "IR Function:\n" << ir_str << "\n";

        // Test 5: Get TVM build config
        std::cout << "\n[Test 5] TVM Build Configuration\n";
        std::cout << "--------------------------------\n";
        auto build_config = TVMFFI::get_tvm_build_config();
        for (const auto& pair : build_config) {
            std::cout << "  " << std::setw(10) << std::left << pair.first << ": "
                      << (pair.second ? "✓ Supported" : "✗ Not supported") << "\n";
        }

        print_separator();
        std::cout << "\n✓ All tests completed successfully!\n";
        print_separator();

        // Cleanup
        PythonHook::finalize();

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}
