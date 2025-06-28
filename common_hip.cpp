#include "common_hip.hpp"
#include <numeric>
#include <random>

// ============================================================================
// GpuTensor Class Implementations
// ============================================================================

GpuTensor::GpuTensor(const std::string& name) : name_(name) {}

GpuTensor::GpuTensor(const std::vector<int>& dimensions, const std::string& name, DataType type)
    : dims_(dimensions), name_(name), dtype(type) {
    allocate(dims_);
}

GpuTensor::~GpuTensor() noexcept {
    try {
        free();
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR in GpuTensor destructor for %s: %s\n", name_.c_str(), e.what());
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

void GpuTensor::allocate(const std::vector<int>& new_dims) {
    if (is_view_) {
        // Views should not re-allocate memory themselves.
        // Detach the view first if reallocation is needed.
        d_ptr_ = nullptr;
        is_view_ = false;
    }

    dims_ = new_dims;
    size_t new_num_elements = 1;
    for (int dim : dims_) {
        if (dim <= 0) {
            new_num_elements = 0;
            break;
        }
        new_num_elements *= dim;
    }

    if (new_num_elements == 0) {
        free();
        num_elements_ = 0;
        return;
    }

    element_size_ = (dtype == DataType::INT32) ? sizeof(int) : sizeof(float);
    size_t new_size_bytes = new_num_elements * element_size_;
    size_t current_size_bytes = allocated_ ? num_elements_ * element_size_ : 0;

    if (new_size_bytes != current_size_bytes) {
        free();
        num_elements_ = new_num_elements;
        HIP_CHECK(hipMalloc(&d_ptr_, new_size_bytes));
        allocated_ = true;
    } else {
        num_elements_ = new_num_elements;
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
    if (allocated_ && d_ptr_ != nullptr && !is_view_) {
        hipError_t err = hipFree(d_ptr_);
        if (err != hipSuccess && err != hipErrorDeinitialized) {
            fprintf(stderr, "HIP Error on hipFree for tensor %s: %s\n", name_.c_str(), hipGetErrorString(err));
        }
    }
    d_ptr_ = nullptr;
    allocated_ = false;
    num_elements_ = 0;
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
    std::vector<int> current_dims = {(int)data.size()};
    allocate(current_dims);

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
    if (weights.is_allocated()) {
        ensure_allocated(grad_weights, weights.dims_);
        grad_weights.name_ = name + "_gw";
    }
    if (has_bias_ && bias.is_allocated()) {
        ensure_allocated(grad_bias, bias.dims_);
        grad_bias.name_ = name + "_gb";
    }
}