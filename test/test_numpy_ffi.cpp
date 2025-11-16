#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "ffi/numpy_ffi.h"
#include "python_hook.h"
#include <vector>
#include <map>
#include <string>

using namespace tvm_sdk::ffi;
using namespace tvm_sdk;
using ::testing::Not;
using ::testing::IsEmpty;
using ::testing::DoubleEq;
using ::testing::DoubleNear;

// Test fixture for NumPyFFI tests
class NumPyFFITest : public ::testing::Test {
protected:
    void SetUp() override {
        // Python initialization is now handled automatically
        // in NumPyFFI conversion functions (vector_to_numpy, etc.)
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// Test: Add Arrays
TEST_F(NumPyFFITest, AddArrays) {
    // Create C++ vectors
    std::vector<double> vec_a = {1.0, 2.0, 3.0};
    std::vector<double> vec_b = {4.0, 5.0, 6.0};

    // Convert to numpy arrays
    auto np_a = NumPyFFI::vector_to_numpy(vec_a);
    auto np_b = NumPyFFI::vector_to_numpy(vec_b);

    // Call Python function
    auto result = NumPyFFI::add_arrays(np_a, np_b);

    // Convert back to C++
    auto result_vec = NumPyFFI::numpy_to_vector<double>(result);

    // Verify results
    ASSERT_EQ(result_vec.size(), 3);
    EXPECT_DOUBLE_EQ(result_vec[0], 5.0);
    EXPECT_DOUBLE_EQ(result_vec[1], 7.0);
    EXPECT_DOUBLE_EQ(result_vec[2], 9.0);

    std::cout << "a + b = [";
    for (size_t i = 0; i < result_vec.size(); i++) {
        std::cout << result_vec[i];
        if (i < result_vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Test: Matrix Multiply
TEST_F(NumPyFFITest, MatrixMultiply) {
    // Create 2D matrices
    std::vector<std::vector<double>> mat_a = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    std::vector<std::vector<double>> mat_b = {
        {5.0, 6.0},
        {7.0, 8.0}
    };

    // Convert to numpy
    auto np_a = NumPyFFI::vector2d_to_numpy(mat_a);
    auto np_b = NumPyFFI::vector2d_to_numpy(mat_b);

    // Matrix multiply
    auto result = NumPyFFI::matrix_multiply(np_a, np_b);

    // Verify result shape
    auto buf = result.request();
    EXPECT_EQ(buf.ndim, 2);
    EXPECT_EQ(buf.shape[0], 2);
    EXPECT_EQ(buf.shape[1], 2);

    // Verify result values
    // A @ B = [[19, 22], [43, 50]]
    double* ptr = static_cast<double*>(buf.ptr);
    EXPECT_DOUBLE_EQ(ptr[0], 19.0);  // [0,0]
    EXPECT_DOUBLE_EQ(ptr[1], 22.0);  // [0,1]
    EXPECT_DOUBLE_EQ(ptr[2], 43.0);  // [1,0]
    EXPECT_DOUBLE_EQ(ptr[3], 50.0);  // [1,1]

    std::cout << "A @ B =" << std::endl;
    for (size_t i = 0; i < 2; i++) {
        std::cout << "  [";
        for (size_t j = 0; j < 2; j++) {
            std::cout << ptr[i * 2 + j];
            if (j < 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

// Test: Create Random Array
TEST_F(NumPyFFITest, CreateRandomArray) {
    std::vector<int> shape = {2, 3};
    auto random_arr = NumPyFFI::create_random_array(shape, 42);

    // Verify shape
    auto buf = random_arr.request();
    EXPECT_EQ(buf.ndim, 2);
    EXPECT_EQ(buf.shape[0], 2);
    EXPECT_EQ(buf.shape[1], 3);

    std::cout << "Random array (2x3) with seed 42:" << std::endl;
    double* ptr = static_cast<double*>(buf.ptr);
    for (size_t i = 0; i < 2; i++) {
        std::cout << "  [";
        for (size_t j = 0; j < 3; j++) {
            std::cout << ptr[i * 3 + j];
            if (j < 2) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

// Test: Array Statistics
TEST_F(NumPyFFITest, ArrayStatistics) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    auto np_arr = NumPyFFI::vector_to_numpy(data);
    auto stats = NumPyFFI::array_statistics(np_arr);

    // Verify statistics are present
    EXPECT_THAT(stats, Not(IsEmpty()));
    EXPECT_TRUE(stats.find("mean") != stats.end());
    EXPECT_TRUE(stats.find("std") != stats.end());
    EXPECT_TRUE(stats.find("min") != stats.end());
    EXPECT_TRUE(stats.find("max") != stats.end());

    // Verify values
    EXPECT_DOUBLE_EQ(stats["mean"], 3.5);
    EXPECT_DOUBLE_EQ(stats["min"], 1.0);
    EXPECT_DOUBLE_EQ(stats["max"], 6.0);
    EXPECT_NEAR(stats["std"], 1.707825127659933, 1e-6);

    std::cout << "Statistics for [1, 2, 3, 4, 5, 6]:" << std::endl;
    std::cout << "  Mean: " << stats["mean"] << std::endl;
    std::cout << "  Std:  " << stats["std"] << std::endl;
    std::cout << "  Min:  " << stats["min"] << std::endl;
    std::cout << "  Max:  " << stats["max"] << std::endl;
}

// Test: Dot Product
TEST_F(NumPyFFITest, DotProduct) {
    std::vector<double> vec_a = {1.0, 2.0, 3.0};
    std::vector<double> vec_b = {4.0, 5.0, 6.0};

    auto np_a = NumPyFFI::vector_to_numpy(vec_a);
    auto np_b = NumPyFFI::vector_to_numpy(vec_b);

    double dot = NumPyFFI::dot_product(np_a, np_b);

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_DOUBLE_EQ(dot, 32.0);

    std::cout << "Dot product of [1, 2, 3] · [4, 5, 6] = " << dot << std::endl;
}
