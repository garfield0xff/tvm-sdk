#ifndef TORCH_FFI_H
#define TORCH_FFI_H

#include "python_hook.h"
#include <string>
#include <map>

namespace tvm_sdk {
namespace ffi {

/**
 * @brief PyTorch FFI (Foreign Function Interface)
 *
 * Provides high-level interface to call PyTorch model operations from C++
 */
class TorchFFI {
public:
    /**
     * @brief Load ResNet18 model with pretrained weights
     * @param pretrained If true, load ImageNet pretrained weights
     * @return Model information as map
     */
    static std::map<std::string, std::string> load_resnet18(bool pretrained = true);

    /**
     * @brief Get ResNet18 model information
     * @return Model information as map (params, architecture details)
     */
    static std::map<std::string, std::string> get_model_info();

    /**
     * @brief Get traced TorchScript model information
     * @param batch_size Batch size for tracing
     * @param height Input image height
     * @param width Input image width
     * @return Traced model information as map
     */
    static std::map<std::string, std::string> get_traced_model_info(
        int batch_size = 1,
        int height = 224,
        int width = 224
    );

    /**
     * @brief Save model state dict
     * @param output_path Path to save state dict
     * @return Save status information as map
     */
    static std::map<std::string, std::string> save_model_state(
        const std::string& output_path = "resnet18_state.pth"
    );

private:
    static constexpr const char* MODULE_PATH = "torch_ext";
};

} // namespace ffi
} // namespace tvm_sdk

#endif // TORCH_FFI_H
