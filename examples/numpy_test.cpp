/**
 * @file numpy_test.cpp
 * @brief Test NumPy FFI functionality
 *
 * Demonstrates how to use NumPyFFI to call Python numpy functions
 * from C++ using the python/numpy/numpy_samples.py module.
 */

#include "ffi/numpy_ffi.h"
#include "python_hook.h"
#include <iostream>
#include <iomanip>

using namespace tvm_sdk::ffi;
using namespace tvm_sdk;

void print_separator() {
    std::cout << "\n========================================\n";
}

void test_add_arrays() {
    std::cout << "\n[Test 1] Add Arrays\n";
    std::cout << "-------------------\n";

    // Create C++ vectors
    std::vector<double> vec_a = {1.0, 2.0, 3.0};
    std::vector<double> vec_b = {4.0, 5.0, 6.0};

    // Convert to numpy arrays
    auto np_a = NumPyFFI::vector_to_numpy(vec_a);
    auto np_b = NumPyFFI::vector_to_numpy(vec_b);

    std::cout << "a = [1.0, 2.0, 3.0]\n";
    std::cout << "b = [4.0, 5.0, 6.0]\n";

    // Call Python function
    auto result = NumPyFFI::add_arrays(np_a, np_b);

    // Convert back to C++
    auto result_vec = NumPyFFI::numpy_to_vector<double>(result);

    std::cout << "a + b = [";
    for (size_t i = 0; i < result_vec.size(); i++) {
        std::cout << result_vec[i];
        if (i < result_vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

void test_matrix_multiply() {
    std::cout << "\n[Test 2] Matrix Multiply\n";
    std::cout << "------------------------\n";

    // Create 2D matrices
    std::vector<std::vector<double>> mat_a = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    std::vector<std::vector<double>> mat_b = {
        {5.0, 6.0},
        {7.0, 8.0}
    };

    std::cout << "A = [[1, 2], [3, 4]]\n";
    std::cout << "B = [[5, 6], [7, 8]]\n";

    // Convert to numpy
    auto np_a = NumPyFFI::vector2d_to_numpy(mat_a);
    auto np_b = NumPyFFI::vector2d_to_numpy(mat_b);

    // Matrix multiply
    auto result = NumPyFFI::matrix_multiply(np_a, np_b);

    std::cout << "A @ B = \n";
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    for (size_t i = 0; i < 2; i++) {
        std::cout << "  [";
        for (size_t j = 0; j < 2; j++) {
            std::cout << std::setw(4) << ptr[i * 2 + j];
            if (j < 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

void test_random_array() {
    std::cout << "\n[Test 3] Create Random Array\n";
    std::cout << "----------------------------\n";

    std::vector<int> shape = {2, 3};
    std::cout << "Shape: (2, 3), Seed: 42\n";

    auto random_arr = NumPyFFI::create_random_array(shape, 42);

    std::cout << "Random array:\n";
    auto buf = random_arr.request();
    double* ptr = static_cast<double*>(buf.ptr);
    for (size_t i = 0; i < 2; i++) {
        std::cout << "  [";
        for (size_t j = 0; j < 3; j++) {
            std::cout << std::fixed << std::setprecision(4) << ptr[i * 3 + j];
            if (j < 2) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

void test_array_statistics() {
    std::cout << "\n[Test 4] Array Statistics\n";
    std::cout << "-------------------------\n";

    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::cout << "Array: [1, 2, 3, 4, 5, 6]\n";

    auto np_arr = NumPyFFI::vector_to_numpy(data);
    auto stats = NumPyFFI::array_statistics(np_arr);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Mean: " << stats["mean"] << "\n";
    std::cout << "  Std:  " << stats["std"] << "\n";
    std::cout << "  Min:  " << stats["min"] << "\n";
    std::cout << "  Max:  " << stats["max"] << "\n";
}

void test_dot_product() {
    std::cout << "\n[Test 5] Dot Product\n";
    std::cout << "--------------------\n";

    std::vector<double> vec_a = {1.0, 2.0, 3.0};
    std::vector<double> vec_b = {4.0, 5.0, 6.0};

    std::cout << "a = [1, 2, 3]\n";
    std::cout << "b = [4, 5, 6]\n";

    auto np_a = NumPyFFI::vector_to_numpy(vec_a);
    auto np_b = NumPyFFI::vector_to_numpy(vec_b);

    double dot = NumPyFFI::dot_product(np_a, np_b);

    std::cout << "a · b = " << dot << "\n";
}

int main() {
    try {
        print_separator();
        std::cout << "NumPy FFI Test Suite\n";
        print_separator();

        // Initialize Python (uses TVM_SDK_PYTHON_PATH environment variable)
        PythonHook::initialize();

        // Run tests
        test_add_arrays();
        test_matrix_multiply();
        test_random_array();
        test_array_statistics();
        test_dot_product();

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
