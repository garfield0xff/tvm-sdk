#include "ffi/tvm_ffi.h"
#include "python_hook.h"
#include <stdexcept>

namespace tvm_sdk {
namespace ffi {

std::string TVMFFI::get_tvm_version() {
    py::object result = PythonHook::call_function(MODULE_NAME, "get_tvm_version");
    return PythonHook::to_cpp<std::string>(result);
}

std::map<std::string, std::string> TVMFFI::get_tvm_target(const std::string& target_name) {
    py::object result = PythonHook::call_function(MODULE_NAME, "get_tvm_target", target_name);
    py::dict dict_result = py::cast<py::dict>(result);

    std::map<std::string, std::string> target_info;
    for (auto item : dict_result) {
        std::string key = py::cast<std::string>(item.first);

        // Handle both string and list values
        if (key == "keys") {
            py::list keys_list = py::cast<py::list>(item.second);
            std::string keys_str = "[";
            for (size_t i = 0; i < py::len(keys_list); i++) {
                if (i > 0) keys_str += ", ";
                keys_str += py::cast<std::string>(keys_list[i]);
            }
            keys_str += "]";
            target_info[key] = keys_str;
        } else {
            target_info[key] = py::cast<std::string>(item.second);
        }
    }

    return target_info;
}

std::map<std::string, bool> TVMFFI::check_tvm_modules() {
    py::object result = PythonHook::call_function(MODULE_NAME, "check_tvm_modules");
    py::dict dict_result = py::cast<py::dict>(result);

    std::map<std::string, bool> modules;
    for (auto item : dict_result) {
        std::string key = py::cast<std::string>(item.first);
        bool value = py::cast<bool>(item.second);
        modules[key] = value;
    }

    return modules;
}

std::string TVMFFI::create_simple_ir() {
    py::object result = PythonHook::call_function(MODULE_NAME, "create_simple_ir");
    return PythonHook::to_cpp<std::string>(result);
}

std::map<std::string, bool> TVMFFI::get_tvm_build_config() {
    py::object result = PythonHook::call_function(MODULE_NAME, "get_tvm_build_config");
    py::dict dict_result = py::cast<py::dict>(result);

    std::map<std::string, bool> config;
    for (auto item : dict_result) {
        std::string key = py::cast<std::string>(item.first);
        bool value = py::cast<bool>(item.second);
        config[key] = value;
    }

    return config;
}

} // namespace ffi
} // namespace tvm_sdk
