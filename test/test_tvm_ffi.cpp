#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "ffi/tvm_ffi.h"
#include "python_hook.h"
#include <string>
#include <map>

using namespace tvm_sdk::ffi;
using namespace tvm_sdk;
using ::testing::Not;
using ::testing::IsEmpty;
using ::testing::ContainsRegex;

// Test fixture for TVMFFI tests
class TVMFFITest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// Test: get_tvm_version should return non-empty version string
TEST_F(TVMFFITest, GetTVMVersion) {
    std::string version = TVMFFI::get_tvm_version();
    EXPECT_THAT(version, Not(IsEmpty()));
    std::cout << "TVM Version: " << version << std::endl;
}

// Test: create_simple_ir should return IR string
TEST_F(TVMFFITest, CreateSimpleIR) {
    py::object result = PythonHook::call_function("tvm_ext.ffi_entry", "create_simple_ir");
    std::string ir = PythonHook::to_cpp<std::string>(result);
    EXPECT_THAT(ir, Not(IsEmpty()));
    std::cout << "Simple IR:\n" << ir << std::endl;
}

// Test: get_tvm_build_config should return build configuration
TEST_F(TVMFFITest, GetTVMBuildConfig) {
    py::object result = PythonHook::call_function("tvm_ext.ffi_entry", "get_tvm_build_config");
    py::dict dict_result = py::cast<py::dict>(result);

    std::map<std::string, bool> config;
    for (auto item : dict_result) {
        std::string key = py::cast<std::string>(item.first);
        bool value = py::cast<bool>(item.second);
        config[key] = value;
    }

    EXPECT_THAT(config, Not(IsEmpty()));

    // Print build config
    std::cout << "TVM Build Config:" << std::endl;
    for (const auto& [target, available] : config) {
        std::cout << "  " << target << ": " << (available ? "enabled" : "disabled") << std::endl;
    }
}

// Test: create_simple_relax_ir should return valid JSON
TEST_F(TVMFFITest, CreateSimpleRelaxIR) {
    py::object result = PythonHook::call_function("tvm_ext.ffi_entry", "create_simple_relax_ir");
    std::string relax_ir = PythonHook::to_cpp<std::string>(result);
    EXPECT_THAT(relax_ir, Not(IsEmpty()));
    std::cout << "Simple Relax IR:\n" << relax_ir << std::endl;
}