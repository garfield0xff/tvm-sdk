/**
 * @file tvm_metaschedule.cpp
 * @brief Test TVM MetaSchedule functionality
 *
 * Demonstrates how to use TVMFFI to perform MetaSchedule auto-tuning
 * from C++ using the python/tvm_ext/metaschedule.py module.
 */

#include "ffi/tvm_ffi.h"
#include "python_hook.h"
#include <iostream>
#include <iomanip>

using namespace tvm_sdk::ffi;
using namespace tvm_sdk;

void print_separator(const std::string& title = "") {
    std::cout << "\n========================================\n";
    if (!title.empty()) {
        std::cout << title << "\n";
        std::cout << "========================================\n";
    }
}

void print_map(const std::map<std::string, std::string>& map_data, const std::string& prefix = "  ") {
    for (const auto& pair : map_data) {
        std::cout << prefix << std::setw(20) << std::left << pair.first << ": " << pair.second << "\n";
    }
}

/**
 * @brief Create a simple Relax IR for testing MetaSchedule
 * @return Relax module IR string
 */
std::string create_test_relax_ir() {
    // This is a simple example IR - in real usage, you would import a model
    // For this demo, we'll use Python to create a simple IR
    py::object result = PythonHook::call_function(
        "tvm_ext.ffi_entry",
        "create_simple_ir"
    );
    return PythonHook::to_cpp<std::string>(result);
}

int main() {
    try {
        print_separator("TVM MetaSchedule Test Suite");

        // Initialize Python
        PythonHook::initialize();

        // Test 1: Get MetaSchedule configuration
        std::cout << "\n[Test 1] MetaSchedule Configuration\n";
        std::cout << "-----------------------------------\n";
        auto ms_config = TVMFFI::get_metaschedule_config();
        print_map(ms_config);

        // Test 2: Check tuning database (should not exist initially)
        std::cout << "\n[Test 2] Check Tuning Database (Initial)\n";
        std::cout << "----------------------------------------\n";
        std::string work_dir = "test_tuning_db";
        auto db_info = TVMFFI::check_tuning_database(work_dir);
        print_map(db_info);

        // Test 3: Get TVM version and build config
        std::cout << "\n[Test 3] TVM Build Configuration\n";
        std::cout << "--------------------------------\n";
        std::string version = TVMFFI::get_tvm_version();
        std::cout << "  TVM Version: " << version << "\n\n";

        // Test 4: Create simple Relax IR for testing
        std::cout << "\n[Test 4] Create Simple Relax IR\n";
        std::cout << "-------------------------------\n";
        std::cout << "Creating a simple matrix multiplication Relax IR...\n";

        std::string relax_ir;
        try {
            py::object result = PythonHook::call_function("tvm_ext.ffi_entry", "create_simple_relax_ir");
            relax_ir = PythonHook::to_cpp<std::string>(result);
            std::cout << "✓ Relax IR created successfully\n";
            std::cout << "  IR size: " << relax_ir.length() << " bytes\n";
        } catch (const std::exception& e) {
            std::cerr << "✗ Failed to create Relax IR: " << e.what() << "\n";
            relax_ir = "";
        }

        // Test 5: Actual MetaSchedule Tuning Test
        std::cout << "\n[Test 5] MetaSchedule Tuning Test\n";
        std::cout << "---------------------------------\n";

        if (!relax_ir.empty()) {
            std::cout << "Running MetaSchedule tuning with small number of trials...\n";
            std::cout << "  Target: llvm\n";
            std::cout << "  Trials: 2 (very small for quick test)\n";
            std::cout << "  Workers: 2\n";
            std::cout << "  Work dir: " << work_dir << "\n\n";

            try {
                std::cout << "Starting tuning (this may take a moment)...\n";
                auto tune_result = TVMFFI::tune_with_metaschedule(
                    relax_ir,
                    "llvm",
                    2,        // Very small number of trials for testing
                    2,        // 2 workers
                    work_dir
                );

                std::cout << "\nTuning completed!\n";
                std::cout << "Results:\n";
                print_map(tune_result);

                if (tune_result["status"] == "success") {
                    std::cout << "\n✓ MetaSchedule tuning successful!\n";
                } else {
                    std::cout << "\n✗ Tuning failed. Check error message above.\n";
                }

            } catch (const std::exception& e) {
                std::cerr << "\n✗ Exception during tuning: " << e.what() << "\n";
            }

            // Test 6: Check tuning database after tuning
            std::cout << "\n[Test 6] Check Tuning Database (After Tuning)\n";
            std::cout << "--------------------------------------------\n";
            auto db_info_after = TVMFFI::check_tuning_database(work_dir);
            print_map(db_info_after);

        } else {
            std::cout << "Skipping tuning test - Relax IR creation failed.\n";
        }

        // Cleanup
        PythonHook::finalize();

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}
