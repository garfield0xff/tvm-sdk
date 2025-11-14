#ifndef TVM_FFI_H
#define TVM_FFI_H

#include "python_hook.h"
#include <string>
#include <map>
#include <vector>

namespace tvm_sdk {
namespace ffi {

/**
 * @brief TVM FFI (Foreign Function Interface)
 *
 * Provides high-level interface to call TVM functions from C++
 * using the python/tvm_ext/tvm_samples.py module.
 */
class TVMFFI {
public:
    /**
     * @brief Get TVM version
     * @return TVM version string
     */
    static std::string get_tvm_version();

    /**
     * @brief Get TVM target information
     * @param target_name Target name (e.g., "llvm", "cuda")
     * @return Target information as map
     */
    static std::map<std::string, std::string> get_tvm_target(const std::string& target_name = "llvm");

    /**
     * @brief Check available TVM modules
     * @return Module availability as map (module_name -> bool)
     */
    static std::map<std::string, bool> check_tvm_modules();

    /**
     * @brief Create a simple TVM IR
     * @return IR string representation
     */
    static std::string create_simple_ir();

    /**
     * @brief Get TVM build configuration
     * @return Build configuration as map (target -> available)
     */
    static std::map<std::string, bool> get_tvm_build_config();

private:
    static constexpr const char* MODULE_NAME = "tvm_ext.tvm_samples";
};

} // namespace ffi
} // namespace tvm_sdk

#endif // TVM_FFI_H
