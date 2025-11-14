# Python Bridge Examples

이 디렉토리는 C++에서 Python을 호출하여 TVM과 상호작용하는 예제를 포함합니다.

## 개요

`PythonBridge` 클래스는 pybind11을 사용하여 C++에서 Python 코드를 실행할 수 있게 해줍니다. 이는 `CPP_TO_PYTHON_CALLS.md`에 설명된 패턴을 따릅니다.

## 주요 기능

- **Python 인터프리터 초기화/종료**
- **TVM 모듈 import**
- **TVM 버전 정보 가져오기**
- **임의의 Python 모듈 import**

## 구현된 패턴 (CPP_TO_PYTHON_CALLS.md 기반)

### 1. GIL (Global Interpreter Lock) 관리
```cpp
py::gil_scoped_acquire gil;  // 자동으로 GIL 획득/해제
```

### 2. Python 모듈 Import
```cpp
py::object tvm = py::module_::import("tvm");
```

### 3. 속성 접근
```cpp
py::object version = tvm.attr("__version__");
```

### 4. Python → C++ 변환
```cpp
std::string version = py::cast<std::string>(version_obj);
```

### 5. 에러 처리
```cpp
try {
    // Python 호출
} catch (const py::error_already_set& e) {
    // Python 예외 처리
}
```

## 예제 파일

### 1. `simple_tvm_import.cpp`
가장 간단한 예제 - TVM 버전만 출력

```cpp
#include "bridge/python_bridge.h"
#include <iostream>

int main() {
    try {
        std::string version = tvm_sdk::bridge::PythonBridge::get_tvm_version();
        std::cout << "TVM Version: " << version << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

### 2. `tvm_version_example.cpp`
완전한 예제 - 초기화, TVM 확인, 다른 모듈 import, 종료

```cpp
// Python 인터프리터 초기화
PythonBridge::initialize();

// TVM 사용 가능 여부 확인
if (PythonBridge::is_tvm_available()) {
    std::string version = PythonBridge::get_tvm_version();
    std::cout << "TVM Version: " << version << std::endl;
}

// 다른 모듈 import
auto numpy = PythonBridge::import_module("numpy");

// 종료
PythonBridge::finalize();
```

## 빌드 방법

### 필요 사항
- C++17 이상
- pybind11 헤더 (`third_party/pybind/`)
- Python 3.x 개발 헤더
- TVM Python 패키지 설치

### 컴파일 예제 (Linux/macOS)

```bash
# Python 버전 확인
python3 --version
python3-config --includes

# 간단한 예제 빌드
g++ -std=c++17 examples/simple_tvm_import.cpp \
    src/bridge/python_bridge.cpp \
    -I./include \
    -I./third_party/pybind \
    $(python3-config --includes) \
    $(python3-config --ldflags) \
    -o simple_tvm_import

# 전체 예제 빌드
g++ -std=c++17 examples/tvm_version_example.cpp \
    src/bridge/python_bridge.cpp \
    -I./include \
    -I./third_party/pybind \
    $(python3-config --includes) \
    $(python3-config --ldflags) \
    -o tvm_version_example
```

### 실행

```bash
# TVM 설치 확인
pip install apache-tvm

# 실행
./simple_tvm_import
# 출력: TVM Version: 0.22.0

./tvm_version_example
# 출력:
# Initializing Python interpreter...
# Checking if TVM is available...
# ✓ TVM is available
# TVM Version: 0.22.0
# ...
```

## API 레퍼런스

### `PythonBridge::initialize()`
Python 인터프리터를 초기화합니다. 이미 초기화된 경우 아무 작업도 하지 않습니다.

### `PythonBridge::finalize()`
Python 인터프리터를 종료합니다.

### `PythonBridge::get_tvm_version()`
TVM 버전을 문자열로 반환합니다.
- **반환값**: `std::string` (예: "0.22.0")
- **예외**: TVM을 import할 수 없으면 `std::runtime_error` 발생

### `PythonBridge::is_tvm_available()`
TVM을 import할 수 있는지 확인합니다.
- **반환값**: `bool` (사용 가능하면 `true`)

### `PythonBridge::import_module(const std::string& module_name)`
Python 모듈을 import합니다.
- **매개변수**: 모듈 이름 (예: "numpy", "torch")
- **반환값**: `py::object` (Python 모듈 객체)
- **예외**: import 실패 시 `std::runtime_error` 발생

## 핵심 디자인 원칙

이 구현은 `CPP_TO_PYTHON_CALLS.md`의 베스트 프랙티스를 따릅니다:

1. **항상 GIL 관리**: `py::gil_scoped_acquire`로 자동 관리
2. **RAII 패턴**: pybind11 객체는 자동으로 레퍼런스 카운트 관리
3. **에러 체크 필수**: 모든 Python 호출에 try-catch 사용
4. **타입 안전**: pybind11의 타입 안전 래퍼 사용

## 참고 자료

- `docks/CPP_TO_PYTHON_CALLS.md` - PyTorch의 C++→Python 호출 패턴 분석
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [Python C API](https://docs.python.org/3/c-api/)

## 문제 해결

### "ModuleNotFoundError: No module named 'tvm'"
```bash
pip install apache-tvm
```

### "ImportError: DLL load failed" (Windows)
Python 버전과 컴파일러 버전이 일치하는지 확인하세요.

### Segmentation Fault
- Python 인터프리터가 초기화되었는지 확인
- GIL이 제대로 획득되었는지 확인
- Python 객체의 레퍼런스 카운트 확인
