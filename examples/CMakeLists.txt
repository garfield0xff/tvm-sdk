# Examples CMakeLists.txt

# TVM MetaSchedule Example
add_executable(tvm_metaschedule tvm_metaschedule.cpp)
target_link_libraries(tvm_metaschedule PRIVATE tvm_sdk_bridge)
target_include_directories(tvm_metaschedule PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${Python3_INCLUDE_DIRS}
)

# Install targets
install(TARGETS tvm_metaschedule
    RUNTIME DESTINATION bin
)

message(STATUS "Examples configured:")
message(STATUS "  - tvm_metaschedule")
