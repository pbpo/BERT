cmake_minimum_required(VERSION 3.20)
project(HIPCANBert LANGUAGES CXX HIP)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- ROCm 경로 설정 ---
if(NOT DEFINED ENV{ROCM_PATH} AND NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
else()
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    endif()
endif()
message(STATUS "Using ROCM_PATH: ${ROCM_PATH}")

# CMake 모듈 및 프리픽스 경로
set(CMAKE_MODULE_PATH ${ROCM_PATH}/hip/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${ROCM_PATH} ${CMAKE_PREFIX_PATH})

# --- CMake 정책 ---
cmake_policy(SET CMP0148 NEW)

# --- 의존 패키지 찾기 ---
find_package(pybind11 REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# --- 소스 파일 목록 ---
set(ALL_SOURCES
    bindings.cpp
    common_hip.cpp
    language_model_hip.cpp
    bert_components_hip.cpp
    attention_hip.cpp
    nn_layers_hip.cpp
    optimizer_hip.cpp
    hip_kernels.cpp
)

# --- Python 확장 모듈 생성 ---
pybind11_add_module(hipbert_core MODULE ${ALL_SOURCES})

# --- 모듈 속성 설정 ---
set_target_properties(hipbert_core PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CXX_VISIBILITY_PRESET "default"
    VISIBILITY_INLINES_HIDDEN OFF
)

# --- HIP 파일 설정 ---
set_source_files_properties(hip_kernels.cpp PROPERTIES LANGUAGE HIP)

# --- 컴파일 옵션 ---
target_compile_options(hipbert_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -fPIC>
    $<$<COMPILE_LANGUAGE:HIP>:-O3 --offload-arch=gfx1030>
)

# --- 인클루드 디렉터리 ---
target_include_directories(hipbert_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${nlohmann_json_SOURCE_DIR}/include
    ${ROCM_PATH}/include
)

# --- 링크 라이브러리 ---
target_link_libraries(hipbert_core PRIVATE
    pybind11::module
    ${rocblas_LIBRARIES}
    ${rocsolver_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# --- 설치 설정 (선택) ---
install(TARGETS hipbert_core
        LIBRARY
        DESTINATION hipbert_core)
