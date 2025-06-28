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

# rocBLAS 찾기
find_package(rocblas REQUIRED)
if(rocblas_FOUND)
    message(STATUS "Found rocBLAS")
else()
    message(FATAL_ERROR "rocBLAS not found! Ensure ROCM_PATH is correct and rocBLAS is installed.")
endif()

# rocSOLVER 찾기
find_package(rocsolver REQUIRED)  # <-- 이름은 소문자 rocsolver가 정확
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
    ${ROCM_PATH}/include
)

# 소스 파일 수집
set(ALL_SOURCES)

# 필수 파일
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    list(APPEND ALL_SOURCES main.cpp)
else()
    message(FATAL_ERROR "main.cpp not found!")
endif()

# 선택적 C++ 파일
set(OPTIONAL_CXX_FILES
    common_hip.cpp
    bert_components_hip.cpp
    attention_hip.cpp
    language_model_hip.cpp
    nn_layers_hip.cpp
    optimizer_hip.cpp
)

foreach(file ${OPTIONAL_CXX_FILES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
        list(APPEND ALL_SOURCES ${file})
        message(STATUS "Added C++ source: ${file}")
    else()
        message(WARNING "C++ source not found: ${file}")
    endif()
endforeach()

# HIP 커널 파일
set(HIP_KERNEL_FILES
    hip_kernels.cpp
)

foreach(file ${HIP_KERNEL_FILES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
        list(APPEND ALL_SOURCES ${file})
        set_source_files_properties(${file} PROPERTIES LANGUAGE HIP)
        message(STATUS "Added HIP kernel source: ${file}")
    else()
        message(WARNING "HIP kernel source not found: ${file}")
    endif()
endforeach()

# 헤더 파일
file(GLOB HEADER_FILES "*.hpp" "*.h")
if(HEADER_FILES)
    list(APPEND ALL_SOURCES ${HEADER_FILES})
    message(STATUS "Added headers: ${HEADER_FILES}")
endif()

# 실행 파일 생성
add_executable(pretrain ${ALL_SOURCES})

# 컴파일 옵션
target_compile_options(pretrain PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:HIP>:-O3 --offload-arch=gfx1030 --offload-arch=gfx1031>
)

# 라이브러리 링크 (수정된 부분)
target_link_libraries(pretrain PRIVATE 
    ${rocblas_LIBRARIES}
    ${rocsolver_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# 디버그 출력
message(STATUS "All Sources: ${ALL_SOURCES}")
message(STATUS "Project configured. Build with 'cmake --build <build_dir>' or 'make' in build dir.")
