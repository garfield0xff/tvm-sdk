"""
TVM Sample Functions for C++ Integration
"""

import tvm
from tvm import relax
from tvm.script import tir as T


def get_tvm_version():
    """
    Get TVM version string.

    Returns:
        str: TVM version
    """
    return tvm.__version__


def get_tvm_target(target_name="llvm"):
    """
    Get TVM target information.

    Args:
        target_name: Target name (e.g., "llvm", "cuda")

    Returns:
        dict: Target information
    """
    target = tvm.target.Target(target_name)
    return {
        "kind": str(target.kind),
        "keys": list(target.keys),
        "str": str(target)
    }


def check_tvm_modules():
    """
    Check available TVM modules.

    Returns:
        dict: Module availability status
    """
    modules = {}

    # Check relax
    try:
        from tvm import relax
        modules["relax"] = True
    except ImportError:
        modules["relax"] = False

    # Check script
    try:
        from tvm import script
        modules["script"] = True
    except ImportError:
        modules["script"] = False

    # Check ir
    try:
        from tvm import ir
        modules["ir"] = True
    except ImportError:
        modules["ir"] = False

    return modules


def create_simple_ir():
    """
    Create a simple TVM IR function.

    Returns:
        str: String representation of the IR
    """
    @T.prim_func
    def simple_add(A: T.Buffer((16,), "float32"),
                    B: T.Buffer((16,), "float32"),
                    C: T.Buffer((16,), "float32")):
        for i in T.serial(16):
            with T.block("compute"):
                vi = T.axis.remap("S", [i])
                C[vi] = A[vi] + B[vi]

    return str(simple_add.script())


def get_tvm_build_config():
    """
    Get TVM build configuration.

    Returns:
        dict: Build configuration info
    """
    config = {}

    # LLVM support
    try:
        config["llvm"] = tvm.target.Target("llvm").check_available()
    except:
        config["llvm"] = False

    # CUDA support
    try:
        config["cuda"] = tvm.target.Target("cuda").check_available()
    except:
        config["cuda"] = False

    # Metal support (macOS)
    try:
        config["metal"] = tvm.target.Target("metal").check_available()
    except:
        config["metal"] = False

    return config


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
