#ifndef COMMON_HIP_HPP
#define COMMON_HIP_HPP
#include <random>   //  ←  추가: mt19937, normal_distribution, uniform_real_distribution
#include <cmath>  
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream> // dims_string()을 위해 추가
#include <algorithm>
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
    private:
    std::string      name_;
    std::vector<int> dims_;
    bool             has_bias_;
    GpuTensor        weights_storage_;
    GpuTensor        bias_storage_;
    GpuTensor        grad_weights_storage_;
    GpuTensor        grad_bias_storage_;
      ///< legacy alias
public:
    // -------------------------------------------------------------
    // Initialisation mode selector
    // -------------------------------------------------------------
    enum class InitType {
        NORMAL,            // 𝒩(0, σ²)               param = σ
        XAVIER_UNIFORM,    // 𝒰(−a, a), a = √6/(fan_in+fan_out)
        XAVIER_NORMAL,     // 𝒩(0, σ²), σ = √2/(fan_in+fan_out)
        CONSTANT           // all values = param
    };

    // -------------------------------------------------------------
    // Constructors (implemented inline; header‑only class)
    // -------------------------------------------------------------
    Parameter() = default;

    Parameter(const std::vector<int>& weight_dims,
              const std::string&      name,
              bool                    has_bias = true)
        : name_(name),
          dims_(weight_dims),
          has_bias_(has_bias),
          weights_storage_(weight_dims, name_ + ".weight"),
          bias_storage_(has_bias ? GpuTensor({weight_dims[0]}, name_ + ".bias")
                                 : GpuTensor()),
          grad_weights_storage_(), grad_bias_storage_(),
          // legacy refs
          weights(weights_storage_), bias(bias_storage_),
          grad_weights(grad_weights_storage_), grad_bias(grad_bias_storage_) {}

    Parameter(const std::vector<int>& weight_dims,
              const std::vector<int>& bias_dims,
              const std::string&      name)
        : name_(name), dims_(weight_dims), has_bias_(!bias_dims.empty()),
          weights_storage_(weight_dims, name_ + ".weight"),
          bias_storage_(has_bias_ ? GpuTensor(bias_dims, name_ + ".bias") : GpuTensor()),
          grad_weights_storage_(), grad_bias_storage_(),
          weights(weights_storage_), bias(bias_storage_),
          grad_weights(grad_weights_storage_), grad_bias(grad_bias_storage_) {}

    // -------------------------------------------------------------
    // Memory helpers
    // -------------------------------------------------------------
    inline void allocate_gradients() {
        if (!grad_weights_storage_.is_allocated()) grad_weights_storage_.allocate(dims_);
        if (has_bias_ && !grad_bias_storage_.is_allocated()) grad_bias_storage_.allocate({dims_[0]});
    }

    inline void zero_gradients(hipStream_t stream = 0) {
        if (grad_weights_storage_.is_allocated()) grad_weights_storage_.zero_out(stream);
        if (has_bias_ && grad_bias_storage_.is_allocated()) grad_bias_storage_.zero_out(stream);
    }

    // -------------------------------------------------------------
    // Initialisation helpers
    // -------------------------------------------------------------
    inline void initialize(InitType type = InitType::NORMAL, float param = 0.02f) {
        if (!weights_storage_.is_allocated()) weights_storage_.allocate(dims_);
        if (has_bias_ && !bias_storage_.is_allocated()) bias_storage_.allocate({dims_[0]});

        std::vector<float> host_w(weights_storage_.num_elements_);
        std::mt19937 gen{std::random_device{}()};

        size_t fan_in = 1, fan_out = 1;
        if (!dims_.empty()) {
            fan_out = dims_[0];
            fan_in  = dims_.size() > 1 ? dims_[1] : dims_[0];
            for (size_t i = 2; i < dims_.size(); ++i) { fan_in *= dims_[i]; fan_out *= dims_[i]; }
        }

        auto fill = [&](auto&& dist){ for (auto &v: host_w) v = dist(gen); };

        switch (type) {
            case InitType::NORMAL:
                fill(std::normal_distribution<float>(0.f, param));
                break;
            case InitType::XAVIER_UNIFORM: {
                float a = std::sqrt(6.f / float(fan_in + fan_out));
                fill(std::uniform_real_distribution<float>(-a, a));
                break; }
            case InitType::XAVIER_NORMAL: {
                float sigma = std::sqrt(2.f / float(fan_in + fan_out));
                fill(std::normal_distribution<float>(0.f, sigma));
                break; }
            case InitType::CONSTANT:
                std::fill(host_w.begin(), host_w.end(), param);
                break;
        }
        weights_storage_.to_gpu(host_w);

        if (has_bias_ && bias_storage_.is_allocated()) {
            std::vector<float> host_b(bias_storage_.num_elements_, 0.f);
            bias_storage_.to_gpu(host_b);
        }
    }

    inline void initialize_random(float /*mean*/, float stddev) { initialize(InitType::NORMAL, stddev); }
    inline void initialize_xavier(bool uniform=true)           { initialize(uniform?InitType::XAVIER_UNIFORM:InitType::XAVIER_NORMAL); }

    // -------------------------------------------------------------
    // Public REFERENCE aliases (legacy‑compatible)
    // -------------------------------------------------------------
    GpuTensor& weights;       ///< same object as weights_storage_
    GpuTensor& bias;
    GpuTensor& grad_weights;
    GpuTensor& grad_bias;

    // -------------------------------------------------------------
    // Modern getters (avoid clash in new code)
    // -------------------------------------------------------------
    const GpuTensor& weights_tensor() const        { return weights_storage_; }
    const GpuTensor& bias_tensor()    const        { return bias_storage_; }
    const GpuTensor& grad_weights_tensor() const   { return grad_weights_storage_; }
    const GpuTensor& grad_bias_tensor()    const   { return grad_bias_storage_; }

    const std::string& name() const { return name_; }
    bool has_bias_flag()      const { return has_bias_; }
    bool& has_bias = has_bias_; 


};

#endif