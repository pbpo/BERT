cmake_minimum_required(VERSION 3.20)
project(HIPCANBert CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17) 
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# ROCm/HIP 경로
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

# HIP 언어 활성화
enable_language(HIP)

# rocBLAS 찾기
find_package(rocblas REQUIRED)
if(rocblas_FOUND)
    message(STATUS "Found rocBLAS: ${rocblas_LIBRARIES}")
else()
    message(FATAL_ERROR "rocBLAS not found! Ensure ROCM_PATH is correct and rocBLAS is installed.")
endif()

# rocSOLVER 찾기
find_package(rocSOLVER REQUIRED)
if(rocSOLVER_FOUND)
    message(STATUS "Found rocSOLVER: ${rocSOLVER_LIBRARIES}")
else()
    message(FATAL_ERROR "rocSOLVER not found! Ensure ROCM_PATH is correct and rocSOLVER is installed.")
endif()

# FetchContent를 통해 nlohmann_json 가져오기
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# 포함 디렉토리
include_directories(
    .
    ${nlohmann_json_SOURCE_DIR}/include
    ${ROCM_PATH}/include
    ${rocblas_INCLUDE_DIRS}
    ${rocSOLVER_INCLUDE_DIRS}
)

# --- 소스 파일 ---
file(GLOB ALL_SOURCES 
    "*.cpp"
    "*.hpp"
)

# --- 실행 파일 ---
add_executable(pretrain ${ALL_SOURCES})
# *** 수정됨: 하드코딩된 타겟 이름 대신 변수를 사용하여 라이브러리 링크 ***
target_link_libraries(pretrain PRIVATE 
    ${rocblas_LIBRARIES} 
    ${rocSOLVER_LIBRARIES} 
    nlohmann_json::nlohmann_json
)

message(STATUS "Project configured. Build with 'cmake --build <build_dir>' or 'make' in build dir.")
