# Examples CMakeLists.txt

# Simple TVM Import Example
add_executable(simple_tvm_import simple_tvm_import.cpp)
target_link_libraries(simple_tvm_import PRIVATE tvm_sdk_bridge)
target_include_directories(simple_tvm_import PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${Python3_INCLUDE_DIRS}
)

# TVM Version Example (Full)
add_executable(tvm_version_example tvm_version_example.cpp)
target_link_libraries(tvm_version_example PRIVATE tvm_sdk_bridge)
target_include_directories(tvm_version_example PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${Python3_INCLUDE_DIRS}
)

# Install targets
install(TARGETS simple_tvm_import tvm_version_example
    RUNTIME DESTINATION bin
)

message(STATUS "Examples configured:")
message(STATUS "  - simple_tvm_import")
message(STATUS "  - tvm_version_example")
