# Examples CMakeLists.txt

# TVM Test Example
add_executable(tvm_test tvm_test.cpp)
target_link_libraries(tvm_test PRIVATE tvm_sdk_bridge)
target_include_directories(tvm_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${Python3_INCLUDE_DIRS}
)

# NumPy Test Example
add_executable(numpy_test numpy_test.cpp)
target_link_libraries(numpy_test PRIVATE tvm_sdk_bridge)
target_include_directories(numpy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${Python3_INCLUDE_DIRS}
)

# Install targets
install(TARGETS tvm_test numpy_test
    RUNTIME DESTINATION bin
)

message(STATUS "Examples configured:")
message(STATUS "  - tvm_test")
message(STATUS "  - numpy_test")
