"""
TVM Version Information
"""

import tvm


def get_tvm_version():
    """
    Get TVM version string.

    Returns:
        str: TVM version
    """
    return tvm.__version__


if __name__ == "__main__":
    print("=== TVM Version Info ===\n")
    version = get_tvm_version()
    print(f"TVM Version: {version}")
