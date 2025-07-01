#include "common_hip.hpp"
#include <numeric>
#include <random>

// ============================================================================
// GpuTensor Class Implementations
// ============================================================================

GpuTensor::GpuTensor(const std::string& name) : name_(name) {}

GpuTensor::GpuTensor(const std::vector<int>& dimensions, const std::string& name, DataType type)
    : dims_(dimensions), name_(name), dtype(type) {
    
    // 디버그 로그 추가
    fprintf(stderr, "[DEBUG] GpuTensor 생성자: name='%s', dims.size()=%zu, dims=[", 
            name.c_str(), dimensions.size());
    for (size_t i = 0; i < dimensions.size(); ++i) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%d", dimensions[i]);
    }
    fprintf(stderr, "]\n");
    
    allocate(dims_);
    
    // 할당 후 차원 재확인
    fprintf(stderr, "[DEBUG] GpuTensor 할당 후: name='%s', dims_.size()=%zu, dims_=[", 
            name_.c_str(), dims_.size());
    for (size_t i = 0; i < dims_.size(); ++i) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%d", dims_[i]);
    }
    fprintf(stderr, "], num_elements=%zu\n", num_elements_);
}

GpuTensor::~GpuTensor() {
    try {
        // 동기화 후 해제
        if (allocated_ && d_ptr_ != nullptr) {
            hipError_t sync_err = hipDeviceSynchronize();
            if (sync_err == hipSuccess) {
                free();
            } else {
                std::cerr << "[WARNING] 동기화 실패, 강제 해제: " << hipGetErrorString(sync_err) << std::endl;
                // 강제 해제 시도
                if (!is_view_) {
                    hipFree(d_ptr_);
                }
                d_ptr_ = nullptr;
                allocated_ = false;
            }
        }
    } catch (...) {
        // 소멸자에서는 예외를 던지지 않음
        std::cerr << "[ERROR] GpuTensor 소멸자에서 예외 발생" << std::endl;
    }
}

GpuTensor::GpuTensor(GpuTensor&& other) noexcept
    : d_ptr_(other.d_ptr_),
      dims_(std::move(other.dims_)),
      num_elements_(other.num_elements_),
      element_size_(other.element_size_),
      name_(std::move(other.name_)),
      allocated_(other.allocated_),
      dtype(other.dtype),
      is_view_(other.is_view_) {
    other.d_ptr_ = nullptr;
    other.allocated_ = false;
    other.is_view_ = false;
    other.num_elements_ = 0;
}

GpuTensor& GpuTensor::operator=(GpuTensor&& other) noexcept {
    if (this != &other) {
        free();
        d_ptr_ = other.d_ptr_;
        dims_ = std::move(other.dims_);
        num_elements_ = other.num_elements_;
        element_size_ = other.element_size_;
        name_ = std::move(other.name_);
        allocated_ = other.allocated_;
        dtype = other.dtype;
        is_view_ = other.is_view_;
        
        other.d_ptr_ = nullptr;
        other.allocated_ = false;
        other.is_view_ = false;
        other.num_elements_ = 0;
    }
    return *this;
}

void GpuTensor::allocate(const std::vector<int>& new_dims)
{
    /***********************
     * 0. 뷰(view) 해제 처리
     ***********************/
    if (is_view_) {
        d_ptr_   = nullptr;  // 실제 메모리는 소유하지 않음
        is_view_ = false;
        allocated_ = false;
        num_elements_ = 0;
    }

    /***********************
     * 1. 차원/용량 계산
     ***********************/
    dims_ = new_dims;
    size_t new_elements = 1;
    for (int d : dims_) {
        if (d <= 0) { 
            // 🚨 수정: 0 이하의 차원이 있으면 에러 발생
            throw std::runtime_error("Invalid dimension " + std::to_string(d) + 
                                   " in tensor '" + name_ + "'. All dimensions must be positive.");
        }
        new_elements *= static_cast<size_t>(d);
    }

    // 🚨 추가: 차원 배열이 비어있는 경우 체크
    if (dims_.empty()) {
        throw std::runtime_error("Empty dimensions vector for tensor '" + name_ + "'");
    }

    // 🚨 추가: 너무 큰 텐서 체크 (메모리 오버플로우 방지)
    const size_t MAX_ELEMENTS = SIZE_MAX / sizeof(float) / 2;  // 안전 마진
    if (new_elements > MAX_ELEMENTS) {
        throw std::runtime_error("Tensor '" + name_ + "' too large: " + 
                               std::to_string(new_elements) + " elements");
    }

    if (new_elements == 0) {               // "빈" 텐서
        free();                            // 혹시 남아 있던 메모리 해제
        return;
    }

    element_size_ = (dtype == DataType::INT32) ? sizeof(int) : sizeof(float);
    size_t new_bytes = new_elements * element_size_;
    size_t current_bytes = allocated_ ? num_elements_ * element_size_ : 0;

    /***********************
     * 2. 재할당 여부 결정
     ***********************/
    if (!allocated_ || new_bytes != current_bytes) {
        free();               // 기존 것 해제(있다면)

        /* 2-a) 메모리 확보 */
        // 🚨 추가: 할당 전 디버그 정보

        
        hipError_t err = hipMalloc(&d_ptr_, new_bytes);
        if (err != hipSuccess) {
            d_ptr_ = nullptr; 
            allocated_ = false; 
            num_elements_ = 0;
            throw std::runtime_error(
                "hipMalloc failed for tensor '" + name_ + "': " +
                hipGetErrorString(err) + " (requested " + 
                std::to_string(new_bytes) + " bytes)");
        }

        // 🚨 추가: 할당된 포인터 검증
        if (d_ptr_ == nullptr) {
            throw std::runtime_error("hipMalloc returned nullptr for tensor '" + name_ + "'");
        }
        
        // 🚨 추가: 포인터 유효성 검증
        if (reinterpret_cast<uintptr_t>(d_ptr_) < 0x1000) {
            hipFree(d_ptr_);  // 잘못된 포인터이므로 해제
            d_ptr_ = nullptr;
            allocated_ = false;
            num_elements_ = 0;
            throw std::runtime_error("hipMalloc returned invalid pointer 0x" + 
                                   std::to_string(reinterpret_cast<uintptr_t>(d_ptr_)) + 
                                   " for tensor '" + name_ + "'");
        }

        allocated_ = true;
        num_elements_ = new_elements;
        
        printf("[DEBUG] %s: hipMalloc 성공 - ptr=%p, bytes=%zu\n", 
               name_.c_str(), d_ptr_, new_bytes);

        /* 2-b) 새 버퍼를 0 으로 초기화(디버깅할 때 유용) */
        err = hipMemset(d_ptr_, 0, new_bytes);
        if (err != hipSuccess) {
            printf("[WARNING] %s: hipMemset 실패: %s\n", 
                   name_.c_str(), hipGetErrorString(err));
            // 초기화 실패는 치명적이지 않으므로 계속 진행
        }

    } else {
        /* 크기는 같지만 재사용 → 단순히 요소 수만 갱신 */
        num_elements_ = new_elements;
        printf("[DEBUG] %s: 메모리 재사용 - ptr=%p, elements=%zu\n", 
               name_.c_str(), d_ptr_, num_elements_);
    }
}
void GpuTensor::allocate_if_smaller(const std::vector<int>& new_dims)
{
    size_t need = GpuTensor::count_elements(new_dims);   // ← 변경
    if (!is_allocated() || need > num_elements_) {
        allocate(new_dims);          // 새 hipMalloc
    } else {
        dims_ = new_dims;            // 메모리는 그대로, 모양만 갱신
    }
}
void GpuTensor::set_view(void* ptr, const std::vector<int>& new_dims, DataType type) {
    free();
    d_ptr_ = ptr;
    dims_ = new_dims;
    is_view_ = true;
    dtype = type;
    element_size_ = (dtype == DataType::INT32) ? sizeof(int) : sizeof(float);
    num_elements_ = 1;
    for (int dim : dims_) {
        num_elements_ *= dim;
    }
    allocated_ = (d_ptr_ != nullptr);
}

void GpuTensor::free() {
    // Only log if name is not empty to reduce noise
    if (!name_.empty()) {
        fprintf(stderr, "[%s] free() 호출됨: allocated_=%s, d_ptr_=%p, is_view_=%s, num_elements_=%zu\n",
                name_.c_str(), allocated_ ? "true" : "false", d_ptr_, is_view_ ? "true" : "false", num_elements_);
    }

    if (allocated_ && d_ptr_ != nullptr && !is_view_) {
        if (!name_.empty()) {
            fprintf(stderr, "[%s] hipFree 호출 시도...\n", name_.c_str());
        }

        // Use hipGetLastError to clear any pending errors before hipFree
        hipGetLastError();
        
        hipError_t err = hipFree(d_ptr_);

        if (!name_.empty()) {
            fprintf(stderr, "[%s] hipFree 실행 완료: 반환 코드(err)=%d, d_ptr_=%p -> nullptr\n",
                    name_.c_str(), err, d_ptr_);
        }

        d_ptr_ = nullptr;
        allocated_ = false;
        num_elements_ = 0;

        if (err != hipSuccess && err != hipErrorDeinitialized) {
            if (!name_.empty()) {
                fprintf(stderr, "!!!!!!!! [%s] hipFree 오류: %s\n", name_.c_str(), hipGetErrorString(err));
            }
        }
    } else {
        // Clean up state for views or already freed tensors
        d_ptr_ = nullptr;
        allocated_ = false;
        num_elements_ = 0;
        is_view_ = false;
    }

    if (!name_.empty()) {
        fprintf(stderr, "[%s] free() 종료 후 상태: allocated_=%s, d_ptr_=%p, num_elements_=%zu\n\n",
                name_.c_str(), allocated_ ? "true" : "false", d_ptr_, num_elements_);
    }
}
void GpuTensor::zero_out(hipStream_t stream) {
    if (allocated_ && d_ptr_ != nullptr && num_elements_ > 0) {
        HIP_CHECK(hipMemsetAsync(d_ptr_, 0, size_in_bytes(), stream));
    }
}

template<typename T>
void GpuTensor::to_gpu(const std::vector<T>& data) {
    if (data.empty()) {
        allocate({0});
        return;
    }
    
    // 기존 차원 정보를 유지하고, 데이터 크기만 확인
    if (dims_.empty()) {
        // 차원이 설정되지 않은 경우에만 1차원으로 설정
        std::vector<int> current_dims = {(int)data.size()};
        allocate(current_dims);
    } else {
        // 기존 차원 정보를 유지하고 메모리만 할당
        size_t expected_elements = 1;
        for (int dim : dims_) {
            expected_elements *= dim;
        }
        
        if (data.size() != expected_elements) {
            throw std::runtime_error("Data size (" + std::to_string(data.size()) + 
                                   ") doesn't match tensor dimensions (" + 
                                   std::to_string(expected_elements) + ") for " + name_);
        }
        
        // 기존 차원을 유지하면서 메모리 할당
        if (!is_allocated() || num_elements_ != expected_elements) {
            allocate(dims_);  // 기존 dims_를 그대로 사용
        }
    }

    if (data.size() != num_elements_) {
        throw std::runtime_error("Tensor size mismatch for " + name_);
    }
    
    HIP_CHECK(hipMemcpy(d_ptr_, data.data(), size_in_bytes(), hipMemcpyHostToDevice));
}

template<typename T>
std::vector<T> GpuTensor::to_cpu() const {
    if (!allocated_ || d_ptr_ == nullptr || num_elements_ == 0) {
        return {};
    }
    std::vector<T> data(num_elements_);
    HIP_CHECK(hipMemcpy(data.data(), d_ptr_, size_in_bytes(), hipMemcpyDeviceToHost));
    return data;
}

void GpuTensor::copy_from_gpu(const GpuTensor& src, hipStream_t stream) {
    if (!src.is_allocated()) {
        throw std::runtime_error("Source tensor " + src.name_ + " is not allocated for copy.");
    }
    allocate(src.dims_);
    if (num_elements_ > 0) {
        HIP_CHECK(hipMemcpyAsync(d_ptr_, src.d_ptr_, src.size_in_bytes(), hipMemcpyDeviceToDevice, stream));
    }
}

// Explicit template instantiations
template void GpuTensor::to_gpu<float>(const std::vector<float>& data);
template std::vector<float> GpuTensor::to_cpu<float>() const;
template void GpuTensor::to_gpu<int>(const std::vector<int>& data);
template std::vector<int> GpuTensor::to_cpu<int>() const;


// ============================================================================
// Parameter Class Implementations
// ============================================================================
Parameter::Parameter(const std::vector<int>& weight_dims, const std::string& name)
    : weights(weight_dims, name + "_w"), has_bias_(false), name(name) {}

Parameter::Parameter(const std::vector<int>& weight_dims, const std::vector<int>& bias_dims, const std::string& name)
    : weights(weight_dims, name + "_w"), bias(bias_dims, name + "_b"), has_bias_(true), name(name) {}

void Parameter::initialize_random(float mean, float stddev) {
        if (!weights.is_allocated())  // ← 기존에 없으면
        weights.allocate(weights.dims_);
    if (!bias.is_allocated() && bias.num_elements_ > 0)
        bias.allocate(bias.dims_);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, stddev);

    if (weights.is_allocated()) {
        std::vector<float> w_data(weights.num_elements_);
        for (auto& val : w_data) val = dist(gen);
        weights.to_gpu(w_data);
    }

    if (has_bias_ && bias.is_allocated()) {
        std::vector<float> b_data(bias.num_elements_, 0.0f);
        bias.to_gpu(b_data);
    }
}

void Parameter::allocate_gradients() {
        if (!grad_weights.is_allocated() && weights.is_allocated())
        grad_weights.allocate(weights.dims_);
    if (bias.num_elements_ && !grad_bias.is_allocated())
        grad_bias.allocate(bias.dims_);
    if (weights.is_allocated()) {
        ensure_allocated(grad_weights, weights.dims_);
        grad_weights.name_ = name + "_gw";
    }
    if (has_bias_ && bias.is_allocated()) {
        ensure_allocated(grad_bias, bias.dims_);
        grad_bias.name_ = name + "_gb";
    }
}