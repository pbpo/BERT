#include "nn_layers_hip.hpp"
#include "hip_kernels.hpp" // For kernel launchers
#include "common_hip.hpp"  // For GpuTensor, Parameter, ROCBLAS_CHECK
#include <stdexcept>       // For runtime_error
#include <vector>
#include <string>
#include <cstdlib> // For std::rand for dropout seed (now replaced)
#include <random>  // For std::random_device, std::mt19937_64

// --- Dropout Implementation ---
// --- Dropout Implementation ---
Dropout::Dropout(float prob) : dropout_prob_(prob) {
    if (prob < 0.0f || prob > 1.0f) {
        throw std::runtime_error("Dropout probability must be between 0 and 1.");
    }
    scale_ = (prob > 0.0f && prob < 1.0f) ? 1.0f / (1.0f - prob) : 1.0f;
}

void Dropout::forward(hipStream_t stream, GpuTensor& input_output, DropoutCache& cache, bool is_training) {
    if (is_training && dropout_prob_ > 0.0f) {
        if (!input_output.is_allocated()) {
            throw std::runtime_error("Input tensor for Dropout::forward is not allocated.");
        }
        cache.mask.allocate(input_output.dims_);

        static std::random_device rd;
        static std::mt19937_64 gen(rd());
        unsigned long long seed = gen();

        launch_dropout_forward(stream,
                               (float*)input_output.d_ptr_,
                               (float*)cache.mask.d_ptr_,
                               (const float*)input_output.d_ptr_,
                               input_output.num_elements_,
                               dropout_prob_,
                               scale_,
                               seed, 0);
    }
}

void Dropout::backward(hipStream_t stream, GpuTensor& grad_input, const GpuTensor& grad_output, const DropoutCache& cache) {
    if (!grad_output.is_allocated()) {
        throw std::runtime_error("grad_output tensor for Dropout::backward is not allocated.");
    }
    if (!grad_input.is_allocated() || grad_input.num_elements_ != grad_output.num_elements_) {
        grad_input.allocate(grad_output.dims_);
    }

    if (dropout_prob_ > 0.0f) {
        if (!cache.mask.is_allocated()) {
            throw std::runtime_error("Dropout mask in cache is not allocated for backward pass.");
        }
        launch_dropout_backward(stream,
                                (float*)grad_input.d_ptr_,
                                (const float*)grad_output.d_ptr_,
                                (const float*)cache.mask.d_ptr_,
                                grad_input.num_elements_,
                                scale_);
    } else {
        grad_input.copy_from_gpu(grad_output, stream);
    }
}
void Dropout::backward(hipStream_t stream, GpuTensor& grad_io, const DropoutCache& cache) {
    // grad_io를 입력과 출력 모두로 사용
    GpuTensor temp_grad_output;
    temp_grad_output.allocate(grad_io.dims_);
    temp_grad_output.copy_from_gpu(grad_io, stream);
    
    // 4개 매개변수 버전 호출
    backward(stream, grad_io, temp_grad_output, cache);
}
// --- DenseLayer Implementation ---
DenseLayer::DenseLayer(const BertConfig& config, int input_dim, int output_dim, const std::string& name_prefix, bool has_bias)
    : weights(has_bias ? Parameter({output_dim, input_dim}, {output_dim}, name_prefix + ".weight") : Parameter({output_dim, input_dim}, name_prefix + ".weight")),
      bias(has_bias ? Parameter({output_dim}, name_prefix + ".bias") : Parameter({0}, name_prefix + ".bias")),
      has_bias_(has_bias) {
    // config 매개변수는 일관성을 위해 유지하지만 사용하지 않음
}

std::vector<Parameter*> DenseLayer::get_parameters() {
    std::vector<Parameter*> params;
    params.push_back(&weights);
    if (has_bias_) {
        params.push_back(&bias);
    }
    return params;
}

void DenseLayer::forward(rocblas_handle blas_handle, hipStream_t stream,
                         const GpuTensor& input, GpuTensor& output,
                         DenseCache& cache, bool is_training) {
    if (!input.is_allocated() || !weights.weights.is_allocated()) {
        throw std::runtime_error("Input or weights tensor not allocated for DenseLayer::forward");
    }

    int in_features = weights.weights.dim_size(1);
    int out_features = weights.weights.dim_size(0);

    if (input.dims_.back() != in_features) {
        throw std::runtime_error("Input feature size mismatch for DenseLayer");
    }

    size_t batch_size_combined = 1;
    std::vector<int> output_dims;
    for (size_t i = 0; i < input.dims_.size() - 1; ++i) {
        batch_size_combined *= input.dim_size(i);
        output_dims.push_back(input.dim_size(i));
    }
    output_dims.push_back(out_features);

    if (!output.is_allocated() || output.dims_ != output_dims) {
        output.allocate(output_dims);
    }

    int M = static_cast<int>(batch_size_combined);
    int K = in_features;
    int N = out_features;

    // 역전파를 위해 입력 저장
    cache.input = &input;

    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));
    const float alpha = 1.0f, beta = 0.0f;

    // 연산: output = input * weights^T
    ROCBLAS_CHECK(rocblas_sgemm(blas_handle,
                                rocblas_operation_none,
                                rocblas_operation_transpose,
                                M, N, K,
                                &alpha,
                                (const float*)input.d_ptr_, K,
                                (const float*)weights.weights.d_ptr_, K,
                                &beta,
                                (float*)output.d_ptr_, N));

    if (has_bias_) {
        launch_add_bias_gelu_kernel(stream,
                                   (float*)output.d_ptr_,
                                   (const float*)output.d_ptr_,
                                   (const float*)bias.weights.d_ptr_,
                                   M, N);
    }
}

void DenseLayer::backward(rocblas_handle blas_handle, hipStream_t stream,
                          const GpuTensor& grad_output, const DenseCache& cache,
                          GpuTensor& grad_input) {
    if (!grad_output.is_allocated() ||
        !cache.input || !cache.input->is_allocated() ||
        !weights.weights.is_allocated()) {
        throw std::runtime_error("Required tensors not allocated for DenseLayer::backward");
    }

    int out_features = weights.weights.dim_size(0);
    int in_features = weights.weights.dim_size(1);

    if (!grad_input.is_allocated() || grad_input.dims_ != cache.input->dims_) {
        grad_input.allocate(cache.input->dims_);
    }
    
    // Allocate gradients
    weights.allocate_gradients();
    if (has_bias_) {
        bias.allocate_gradients();
    }

    size_t batch_size_combined = 1;
    for (size_t i = 0; i < grad_output.dims_.size() - 1; ++i) {
        batch_size_combined *= grad_output.dim_size(i);
    }

    int M = static_cast<int>(batch_size_combined);
    int N = out_features;
    int K = in_features;

    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));
    const float alpha = 1.0f, beta = 0.0f;

    // Calculate grad_input = grad_output * weights
    ROCBLAS_CHECK(rocblas_sgemm(blas_handle,
                                rocblas_operation_none,
                                rocblas_operation_none,
                                M, K, N,
                                &alpha,
                                (const float*)grad_output.d_ptr_, N,
                                (const float*)weights.weights.d_ptr_, K,
                                &beta,
                                (float*)grad_input.d_ptr_, K));

    // Calculate grad_weights = input^T * grad_output
    ROCBLAS_CHECK(rocblas_sgemm(blas_handle,
                                rocblas_operation_transpose,
                                rocblas_operation_none,
                                K, N, M,
                                &alpha,
                                (const float*)cache.input->d_ptr_, K,
                                (const float*)grad_output.d_ptr_, N,
                                &beta,
                                (float*)weights.grad_weights.d_ptr_, N));

    // Calculate grad_bias if needed
    if (has_bias_) {
        launch_reduce_sum_kernel(stream,
                                (float*)bias.grad_weights.d_ptr_,
                                (const float*)grad_output.d_ptr_,
                                M, N);
    }
}
void DenseLayer::allocate_gradients() {
    weights.allocate_gradients();
    if (has_bias_) {
        bias.allocate_gradients();
    }
}

// --- LayerNorm Implementation ---
LayerNorm::LayerNorm(const BertConfig& config, const std::string& name_prefix)
    : gamma({config.hidden_size}, name_prefix + ".weight"),
      beta({config.hidden_size}, name_prefix + ".bias"),
      eps_(1e-12f) {}

LayerNorm::LayerNorm(int hidden_size, float eps, const std::string& name_prefix)
    : gamma({hidden_size}, name_prefix + ".weight"),
      beta({hidden_size}, name_prefix + ".bias"),
      eps_(eps) {}

std::vector<Parameter*> LayerNorm::get_parameters() {
    return {&gamma, &beta};
}

void LayerNorm::forward(hipStream_t stream, const GpuTensor& input, GpuTensor& output, LayerNormCache& cache) {
    if (!input.is_allocated() || !gamma.weights.is_allocated() || !beta.weights.is_allocated()) {
        throw std::runtime_error("Input or parameters not allocated for LayerNorm::forward");
    }

    int hidden_size = gamma.weights.dim_size(0);
    if (input.dims_.back() != hidden_size) {
        throw std::runtime_error("LayerNorm input's last dimension mismatch");
    }
    
    if (!output.is_allocated() || output.dims_ != input.dims_) {
        output.allocate(input.dims_);
    }

    size_t B_rows = 1;
    for (size_t i = 0; i < input.dims_.size() - 1; ++i) {
        B_rows *= input.dim_size(i);
    }
    int C_cols = hidden_size;

    cache.input = &input;
    cache.normalized_input.allocate(input.dims_);

    launch_layer_norm_forward_optimized(stream,
                              (float*)output.d_ptr_,
                              nullptr, // mean
                              nullptr, // rstd
                              (const float*)input.d_ptr_,
                              (const float*)gamma.weights.d_ptr_,
                              (const float*)beta.weights.d_ptr_,
                              static_cast<int>(B_rows), C_cols, eps_);
                              
    // Store normalized input for backward
   cache.normalized_input.copy_from_gpu(output, stream);

}
void LayerNorm::allocate_gradients() {
    gamma.allocate_gradients();
    beta.allocate_gradients();
}
void LayerNorm::backward(hipStream_t stream, const GpuTensor& grad_output, const LayerNormCache& cache, GpuTensor& grad_input) {
    if (!grad_output.is_allocated() || !cache.input || !cache.input->is_allocated()) {
        throw std::runtime_error("Required tensors not allocated for LayerNorm::backward");
    }

    if (!grad_input.is_allocated() || grad_input.dims_ != grad_output.dims_) {
        grad_input.allocate(grad_output.dims_);
    }
    
    // Allocate gradients
    gamma.allocate_gradients();
    beta.allocate_gradients();

    size_t B_rows = 1;
    for (size_t i = 0; i < cache.input->dims_.size() - 1; ++i) {
        B_rows *= cache.input->dim_size(i);
    }
    int C_cols = cache.input->dims_.back();

    launch_layer_norm_backward_optimized(stream,
                               (float*)grad_input.d_ptr_,
                               (float*)gamma.grad_weights.d_ptr_,
                               (float*)beta.grad_weights.d_ptr_,
                               (const float*)grad_output.d_ptr_,
                               (const float*)cache.input->d_ptr_,
                               (const float*)gamma.weights.d_ptr_,
                               nullptr, // mean
                               nullptr, // rstd
                               static_cast<int>(B_rows), C_cols);
}

// --- Gelu Implementation ---
void Gelu::forward(hipStream_t stream, const GpuTensor& input, GpuTensor& output, GeluCache& cache) {
    if (!input.is_allocated()) {
        throw std::runtime_error("Input tensor not allocated for Gelu::forward.");
    }
    if (!output.is_allocated() || output.dims_ != input.dims_) {
        output.allocate(input.dims_);
    }
    cache.input = &input;
    launch_gelu_forward_kernel(
        stream,
        (float*)output.d_ptr_,
        (const float*)input.d_ptr_,
        input.num_elements_);
}

void Gelu::backward(hipStream_t stream, const GpuTensor& grad_output, GpuTensor& grad_input, const GeluCache& cache) {
    if (!grad_output.is_allocated() || !cache.input || !cache.input->is_allocated()) {
        throw std::runtime_error("Required tensors not allocated for Gelu::backward.");
    }
    if (!grad_input.is_allocated() || grad_input.dims_ != grad_output.dims_) {
        grad_input.allocate(grad_output.dims_);
    }

    launch_gelu_backward_kernel(stream,
                               (float*)grad_input.d_ptr_,
                               (const float*)grad_output.d_ptr_,
                               (const float*)cache.input->d_ptr_,
                               grad_input.num_elements_);
}