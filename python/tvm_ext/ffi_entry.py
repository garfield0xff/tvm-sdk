"""
TVM FFI Entry Point for C++ Integration

This module serves as the main entry point for C++ FFI calls.
It re-exports functions from version, config, and metaschedule modules.
"""

# Import from version module
from .version import get_tvm_version

# Import from config module
from .config import (
    get_tvm_target,
    check_tvm_modules,
    create_simple_ir,
    get_tvm_build_config
)

# Import from metaschedule module
from .metaschedule import (
    tune_with_metaschedule,
    apply_tuning_database,
    compile_with_metaschedule,
    create_simple_relax_ir,
    get_metaschedule_config,
    check_tuning_database
)

# Expose all functions
__all__ = [
    # Version
    'get_tvm_version',
    # Config
    'get_tvm_target',
    'check_tvm_modules',
    'create_simple_ir',
    'get_tvm_build_config',
    # MetaSchedule
    'tune_with_metaschedule',
    'apply_tuning_database',
    'compile_with_metaschedule',
    'create_simple_relax_ir',
    'get_metaschedule_config',
    'check_tuning_database'
]


if __name__ == "__main__":
    print("=== TVM Sample Functions Test ===\n")

    # Test get_tvm_version
    version = get_tvm_version()
    print(f"TVM Version: {version}")

    # Test get_tvm_target
    target_info = get_tvm_target("llvm")
    print(f"\nTarget Info: {target_info}")

    # Test check_tvm_modules
    modules = check_tvm_modules()
    print(f"\nAvailable Modules: {modules}")

    # Test create_simple_ir
    ir_str = create_simple_ir()
    print(f"\nSimple IR:\n{ir_str}")

    # Test get_tvm_build_config
    build_config = get_tvm_build_config()
    print(f"\nBuild Config: {build_config}")
