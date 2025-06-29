cmake_minimum_required(VERSION 3.20)
project(HIPCANBert CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# ROCm/HIP 경로 설정
if(NOT DEFINED ENV{ROCM_PATH} AND NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
    message(STATUS "ROCM_PATH not set, defaulting to /opt/rocm. Override if necessary.")
else()
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    endif()
    message(STATUS "Using ROCM_PATH: ${ROCM_PATH}")
endif()

set(CMAKE_MODULE_PATH ${ROCM_PATH}/hip/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${ROCM_PATH} ${CMAKE_PREFIX_PATH})

enable_language(HIP)

# pybind11 찾기 (추가된 부분)
find_package(pybind11 REQUIRED)

# rocBLAS 찾기
find_package(rocblas REQUIRED)
if(rocblas_FOUND)
    message(STATUS "Found rocBLAS")
else()
    message(FATAL_ERROR "rocBLAS not found! Ensure ROCM_PATH is correct and rocBLAS is installed.")
endif()

# rocSOLVER 찾기
find_package(rocsolver REQUIRED)
if(rocsolver_FOUND)
    message(STATUS "Found rocSOLVER")
else()
    message(FATAL_ERROR "rocSOLVER not found! Ensure ROCM_PATH is correct and rocSOLVER is installed.")
endif()

# JSON 라이브러리 가져오기
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# include 디렉토리
include_directories(
    .
    ${nlohmann_json_SOURCE_DIR}/include
    ${pybind11_INCLUDE_DIRS} # pybind11 헤더 경로 추가
    ${ROCM_PATH}/include
)

# 소스 파일 수집
set(ALL_SOURCES)

set(CXX_FILES
    common_hip.cpp
    bert_components_hip.cpp
    attention_hip.cpp
    language_model_hip.cpp
    nn_layers_hip.cpp
    optimizer_hip.cpp
    workspace_manager.cpp # 누락된 파일 추가
)

foreach(file ${CXX_FILES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
        list(APPEND ALL_SOURCES ${file})
        message(STATUS "Added C++ source: ${file}")
    else()
        message(WARNING "C++ source not found: ${file}")
    endif()
endforeach()

# HIP 커널 파일
set(HIP_KERNEL_FILES hip_kernels.cpp)
foreach(file ${HIP_KERNEL_FILES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
        list(APPEND ALL_SOURCES ${file})
        set_source_files_properties(${file} PROPERTIES LANGUAGE HIP)
        message(STATUS "Added HIP kernel source: ${file}")
    else()
        message(WARNING "HIP kernel source not found: ${file}")
    endif()
endforeach()

# Python 모듈 생성 (수정된 부분)
pybind11_add_module(hipbert_core bindings.cpp ${ALL_SOURCES})


# 컴파일 옵션 (수정된 부분)
target_compile_options(hipbert_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -fPIC> # -fPIC 옵션 추가
    $<$<COMPILE_LANGUAGE:HIP>:-O3 --offload-arch=gfx1030 --offload-arch=gfx1031>
)

# 라이브러리 링크 (수정된 부분)
target_link_libraries(hipbert_core PRIVATE
    ${rocblas_LIBRARIES}
    ${rocsolver_LIBRARIES}
    nlohmann_json::nlohmann_json
)

message(STATUS "Project configured to build 'hipbert_core' Python module.")
message(STATUS "Build with 'cmake --build <build_dir>' or 'make' in build dir.")