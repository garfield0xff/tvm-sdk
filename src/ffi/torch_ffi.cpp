#include "ffi/torch_ffi.h"
#include "python_hook.h"
#include <stdexcept>

namespace tvm_sdk {
namespace ffi {

std::map<std::string, std::string> TorchFFI::load_resnet18(bool pretrained) {
    py::object result = PythonHook::call_function(MODULE_PATH, "load_resnet18", pretrained);

    // Get model info after loading
    py::object info_result = PythonHook::call_function(MODULE_PATH, "get_model_info", result);
    py::dict dict_result = py::cast<py::dict>(info_result);

    std::map<std::string, std::string> model_info;
    for (auto item : dict_result) {
        std::string key = py::cast<std::string>(item.first);

        if (py::isinstance<py::str>(item.second)) {
            model_info[key] = py::cast<std::string>(item.second);
        } else if (py::isinstance<py::int_>(item.second)) {
            model_info[key] = std::to_string(py::cast<int>(item.second));
        } else {
            model_info[key] = py::cast<std::string>(py::str(item.second));
        }
    }

    model_info["pretrained"] = pretrained ? "true" : "false";
    model_info["status"] = "loaded";

    return model_info;
}

std::map<std::string, std::string> TorchFFI::get_model_info() {
    py::object result = PythonHook::call_function(MODULE_PATH, "get_model_info");
    py::dict dict_result = py::cast<py::dict>(result);

    std::map<std::string, std::string> model_info;
    for (auto item : dict_result) {
        std::string key = py::cast<std::string>(item.first);

        if (py::isinstance<py::str>(item.second)) {
            model_info[key] = py::cast<std::string>(item.second);
        } else if (py::isinstance<py::int_>(item.second)) {
            model_info[key] = std::to_string(py::cast<int>(item.second));
        } else {
            model_info[key] = py::cast<std::string>(py::str(item.second));
        }
    }

    return model_info;
}

std::map<std::string, std::string> TorchFFI::get_traced_model_info(
    int batch_size,
    int height,
    int width
) {
    // Create input shape tuple
    py::tuple input_shape = py::make_tuple(batch_size, 3, height, width);

    py::object result = PythonHook::call_function(
        MODULE_PATH,
        "get_traced_model",
        input_shape
    );

    // Result is a tuple of (traced_model, example_input)
    py::tuple result_tuple = py::cast<py::tuple>(result);

    std::map<std::string, std::string> traced_info;
    traced_info["status"] = "traced";
    traced_info["input_shape"] = "(" + std::to_string(batch_size) + ", 3, " +
                                 std::to_string(height) + ", " + std::to_string(width) + ")";
    traced_info["format"] = "torchscript";
    traced_info["method"] = "trace";

    return traced_info;
}

std::map<std::string, std::string> TorchFFI::save_model_state(
    const std::string& output_path
) {
    py::object result = PythonHook::call_function(
        MODULE_PATH,
        "save_model_state",
        output_path
    );

    py::dict dict_result = py::cast<py::dict>(result);

    std::map<std::string, std::string> save_info;
    for (auto item : dict_result) {
        std::string key = py::cast<std::string>(item.first);

        if (py::isinstance<py::str>(item.second)) {
            save_info[key] = py::cast<std::string>(item.second);
        } else {
            save_info[key] = py::cast<std::string>(py::str(item.second));
        }
    }

    return save_info;
}

} // namespace ffi
} // namespace tvm_sdk
