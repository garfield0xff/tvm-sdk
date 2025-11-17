cmake_minimum_required(VERSION 3.15)
project(tvm-sdk VERSION 0.1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build tests" OFF)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python3 found: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${Python3_INCLUDE_DIRS}
)

# Source files
set(BRIDGE_SOURCES
    src/python_hook.cpp
    src/ffi/numpy_ffi.cpp
    src/ffi/tvm_ffi.cpp
    src/ffi/torch_ffi.cpp
)

# Define Python path for the project
set(TVM_SDK_PYTHON_PATH "${CMAKE_SOURCE_DIR}/python")
message(STATUS "TVM SDK Python path: ${TVM_SDK_PYTHON_PATH}")

# Create library
add_library(tvm_sdk_bridge STATIC ${BRIDGE_SOURCES})
target_include_directories(tvm_sdk_bridge PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${Python3_INCLUDE_DIRS}
)
target_link_libraries(tvm_sdk_bridge PUBLIC
    ${Python3_LIBRARIES}
)
target_compile_definitions(tvm_sdk_bridge PRIVATE
    TVM_SDK_PYTHON_PATH="${TVM_SDK_PYTHON_PATH}"
)

# Set compile options
target_compile_options(tvm_sdk_bridge PRIVATE
    -Wall
    -Wextra
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3>
)

# Build examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Build tests
if(BUILD_TESTS)
    add_subdirectory(test)
endif()

# Print configuration
message(STATUS "========================================")
message(STATUS "TVM SDK Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "========================================")
