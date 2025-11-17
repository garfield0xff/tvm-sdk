cmake_minimum_required(VERSION 3.15)

# Define project directory for test execution
set(PROJECT_DIRECTORY ${CMAKE_SOURCE_DIR})

# Enable testing
enable_testing()

# Download and configure GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${Python3_INCLUDE_DIRS}
)

# TVM FFI Test executable
add_executable(test_tvm_ffi
    test_tvm_ffi.cpp
)

# Link libraries
target_link_libraries(test_tvm_ffi
    PRIVATE
    gtest_main
    gmock_main
    tvm_sdk_bridge
    ${Python3_LIBRARIES}
)

# Set compile options
target_compile_options(test_tvm_ffi PRIVATE
    -Wall
    -Wextra
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3>
)

# Pass Python path to test
target_compile_definitions(test_tvm_ffi PRIVATE
    TVM_SDK_PYTHON_PATH="${TVM_SDK_PYTHON_PATH}"
)

# Add test to CTest
include(GoogleTest)
gtest_discover_tests(test_tvm_ffi
    WORKING_DIRECTORY ${PROJECT_DIRECTORY}
)

# NumPy FFI Test executable
add_executable(test_numpy_ffi
    test_numpy_ffi.cpp
)

# Link libraries
target_link_libraries(test_numpy_ffi
    PRIVATE
    gtest_main
    gmock_main
    tvm_sdk_bridge
    ${Python3_LIBRARIES}
)

# Set compile options
target_compile_options(test_numpy_ffi PRIVATE
    -Wall
    -Wextra
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3>
)

# Pass Python path to test
target_compile_definitions(test_numpy_ffi PRIVATE
    TVM_SDK_PYTHON_PATH="${TVM_SDK_PYTHON_PATH}"
)

# Add test to CTest
gtest_discover_tests(test_numpy_ffi
    WORKING_DIRECTORY ${PROJECT_DIRECTORY}
)

# Torch FFI Test executable
add_executable(test_torch_ffi
    test_torch_ffi.cpp
)

# Link libraries
target_link_libraries(test_torch_ffi
    PRIVATE
    gtest_main
    gmock_main
    tvm_sdk_bridge
    ${Python3_LIBRARIES}
)

# Set compile options
target_compile_options(test_torch_ffi PRIVATE
    -Wall
    -Wextra
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3>
)

# Pass Python path and project directory to test
target_compile_definitions(test_torch_ffi PRIVATE
    TVM_SDK_PYTHON_PATH="${TVM_SDK_PYTHON_PATH}"
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

# Add test to CTest
gtest_discover_tests(test_torch_ffi
    WORKING_DIRECTORY ${PROJECT_DIRECTORY}
)

# TVM Schedule Test executable
add_executable(test_tvm_schedule
    test_tvm_schedule.cpp
)

# Link libraries
target_link_libraries(test_tvm_schedule
    PRIVATE
    gtest_main
    gmock_main
    tvm_sdk_bridge
    ${Python3_LIBRARIES}
)

# Set compile options
target_compile_options(test_tvm_schedule PRIVATE
    -Wall
    -Wextra
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3>
)

# Pass Python path and project directory to test
target_compile_definitions(test_tvm_schedule PRIVATE
    TVM_SDK_PYTHON_PATH="${TVM_SDK_PYTHON_PATH}"
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

# Add test to CTest
gtest_discover_tests(test_tvm_schedule
    WORKING_DIRECTORY ${PROJECT_DIRECTORY}
)

# Custom test target for easier execution
add_custom_target(run_tests
    COMMAND test_tvm_ffi && test_numpy_ffi && test_torch_ffi && test_tvm_schedule
    DEPENDS test_tvm_ffi test_numpy_ffi test_torch_ffi test_tvm_schedule
    WORKING_DIRECTORY ${PROJECT_DIRECTORY}
    COMMENT "Running all tests..."
)
