#ifndef COMMON_HIP_HPP
#define COMMON_HIP_HPP

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream> // dims_string()을 위해 추가

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// --- 유틸리티 매크로 ---
#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s in %s at line %d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        throw std::runtime_error("HIP error"); \
    } \
} while(0)

#define ROCBLAS_CHECK(cmd) do { \
    rocblas_status err = cmd; \
    if (err != rocblas_status_success) { \
        fprintf(stderr, "rocBLAS Error: %s (%d) in %s at line %d\n", rocblas_status_to_string(err), err, __FILE__, __LINE__); \
        throw std::runtime_error("rocBLAS error"); \
    } \
} while(0)

// --- 데이터 타입 ---
enum class DataType { FLOAT32, INT32 };

// --- 전방 선언 ---
class Parameter;
struct BertConfig;

// --- GpuTensor: GPU 메모리를 관리하는 텐서 클래스 ---
class GpuTensor {
public:
    void* d_ptr_ = nullptr;
    std::vector<int> dims_;
    size_t num_elements_ = 0;
    size_t element_size_ = 0;
    std::string name_;
    bool allocated_ = false;
    DataType dtype = DataType::FLOAT32;
    bool is_view_ = false;
void allocate_if_smaller(const std::vector<int>& new_dims);
    GpuTensor(const std::string& name = "");
    GpuTensor(const std::vector<int>& dimensions, const std::string& name = "", DataType type = DataType::FLOAT32);
    ~GpuTensor() noexcept; // 소멸자에서 예외가 발생하지 않도록 noexcept 명시
    static size_t count_elements(const std::vector<int>& dims) {
        size_t n = 1;
        for (int d : dims) n *= static_cast<size_t>(d);
        return n;
    }
    GpuTensor(const GpuTensor&) = delete;
    GpuTensor& operator=(const GpuTensor&) = delete;

    GpuTensor(GpuTensor&& other) noexcept;
    GpuTensor& operator=(GpuTensor&& other) noexcept;

    void allocate(const std::vector<int>& new_dims);
    void set_view(void* ptr, const std::vector<int>& dims, DataType type = DataType::FLOAT32);
    void free();
    void zero_out(hipStream_t stream = 0);

    template<typename T>
    void to_gpu(const std::vector<T>& data);
    template<typename T>
    std::vector<T> to_cpu() const;
    void copy_from_gpu(const GpuTensor& src, hipStream_t stream = 0);

    size_t size_in_bytes() const { return num_elements_ * element_size_; }
    bool is_allocated() const { return allocated_; }
    int dim_size(int i) const { return dims_.at(i); }

    /**
     * @brief [추가된 함수] 텐서의 차원 정보를 문자열로 반환합니다.
     * @return "[dim1, dim2, ...]" 형식의 문자열
     */
    std::string dims_string() const {
        if (dims_.empty()) {
            return "[]";
        }
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < dims_.size(); ++i) {
            ss << dims_[i] << (i == dims_.size() - 1 ? "" : ", ");
        }
        ss << "]";
        return ss.str();
    }
};

/**
 * @brief 텐서가 할당되었는지 확인하고, 필요 시 새로 할당하는 헬퍼 함수.
 * @param t 검사할 GpuTensor.
 * @param new_dims 원하는 텐서 차원.
 * @param stream HIP 스트림.
 * @param zero 초기화 여부.
 */
inline void ensure_allocated(GpuTensor& t, const std::vector<int>& new_dims, bool zero = false, hipStream_t stream = 0) {
    bool needs_alloc = !t.is_allocated() || t.dims_ != new_dims;
    if (needs_alloc) {
        t.allocate(new_dims);
    }
    if (zero && t.is_allocated()) {
        t.zero_out(stream);
    }
}

// --- Parameter: 학습 가능한 가중치와 기울기를 관리하는 클래스 ---
class Parameter {
public:
    GpuTensor weights;
    GpuTensor bias;
    GpuTensor grad_weights;
    GpuTensor grad_bias;
    bool has_bias_;
    std::string name;

    Parameter(const std::vector<int>& weight_dims, const std::string& name);
    Parameter(const std::vector<int>& weight_dims, const std::vector<int>& bias_dims, const std::string& name);

    void initialize_random(float mean = 0.0f, float stddev = 0.02f);
    void allocate_gradients();
};

#endif // COMMON_HIP_HPP