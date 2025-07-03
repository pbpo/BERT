// nn_layers_hip.cpp

#include "nn_layers_hip.hpp"
#include "hip_kernels.hpp" // For kernel launchers
#include "common_hip.hpp"  // For GpuTensor, Parameter, ROCBLAS_CHECK
#include <stdexcept>       // For std::runtime_error
#include <vector>
#include <string>
#include <cstdlib>         // For std::rand (legacy)
#include <random>          // For std::random_device, std::mt19937_64

// --- Dropout Implementation ---
Dropout::Dropout(float prob) : dropout_prob_(prob) {
    if (prob < 0.0f || prob > 1.0f) {
        throw std::runtime_error("Dropout probability must be between 0 and 1.");
    }
    scale_ = (prob > 0.0f && prob < 1.0f) ? 1.0f / (1.0f - prob) : 1.0f;
}

void Dropout::forward(hipStream_t stream,
                      GpuTensor& input_output,
                      DropoutCache& cache,
                      bool is_training) {
    if (is_training && dropout_prob_ > 0.0f) {
        if (!input_output.is_allocated()) {
            throw std::runtime_error("Input tensor for Dropout::forward is not allocated.");
        }
        cache.mask.allocate(input_output.dims_);

        static std::random_device rd;
        static std::mt19937_64 gen(rd());
        unsigned long long seed = gen();

        launch_dropout_forward(
            stream,
            (float*)input_output.d_ptr_,
            (float*)cache.mask.d_ptr_,
            (const float*)input_output.d_ptr_,
            input_output.num_elements_,
            dropout_prob_,
            scale_,
            seed,
            0
        );
    }
}

void Dropout::backward(hipStream_t stream,
                       GpuTensor& grad_input,
                       const GpuTensor& grad_output,
                       const DropoutCache& cache) {
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
        launch_dropout_backward(
            stream,
            (float*)grad_input.d_ptr_,
            (const float*)grad_output.d_ptr_,
            (const float*)cache.mask.d_ptr_,
            grad_input.num_elements_,
            scale_
        );
    } else {
        grad_input.copy_from_gpu(grad_output, stream);
    }
}



// --- DenseLayer Implementation ---
DenseLayer::DenseLayer(const BertConfig& config,
                       int input_dim,
                       int output_dim,
                       const std::string& name_prefix,
                       bool has_bias)
    : weights(has_bias
              ? Parameter({output_dim, input_dim}, {output_dim}, name_prefix + ".weight")
              : Parameter({output_dim, input_dim}, name_prefix + ".weight")),
      bias(has_bias
           ? Parameter({output_dim}, name_prefix + ".bias")
           : Parameter({0}, name_prefix + ".bias")),
      has_bias_(has_bias) {}

std::vector<Parameter*> DenseLayer::get_parameters() {
    std::vector<Parameter*> params;
    params.push_back(&weights);
    if (has_bias_) {
        params.push_back(&bias);
    }
    return params;
}

void DenseLayer::forward(rocblas_handle blas_handle,
                         hipStream_t stream,
                         const GpuTensor& input,
                         GpuTensor& output,
                         DenseCache& cache,
                         bool /*is_training*/) {
    // 편의용 매크로: num_elements_ 검사
    #define ASSERT_SAME_NUMEL(a,b,msg) \
        if ((a).num_elements_ != (b).num_elements_) \
            throw std::runtime_error("Size mismatch: " msg)

    if (!input.is_allocated() || !weights.weights.is_allocated()) {
        throw std::runtime_error("Input or weights tensor not allocated for DenseLayer::forward");
    }

    int in_features  = weights.weights.dim_size(1);
    int out_features = weights.weights.dim_size(0);
    if (input.dims_.back() != in_features) {
        throw std::runtime_error("Input feature size mismatch for DenseLayer");
    }

    // 배치 크기 계산 (입력 텐서의 마지막 차원 제외)
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

    // 빈 배치나 차원이 0인 경우 일부러 SGEMM 스킵
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    cache.input = &input;

    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));
    const float alpha = 1.0f, beta = 0.0f;

    // output = input(M×K) * weights^T(K×N)  →  (M×N)
    ROCBLAS_CHECK(rocblas_sgemm(
        blas_handle,
        rocblas_operation_none,    // A no-transpose
        rocblas_operation_transpose, // B transpose
        M, N, K,
        &alpha,
        (const float*)input.d_ptr_,  M,
        (const float*)weights.weights.d_ptr_, N,
        &beta,
        (float*)output.d_ptr_,     M
    ));

    if (has_bias_) {
        launch_add_bias_gelu_kernel(
            stream,
            (float*)output.d_ptr_,
            (const float*)output.d_ptr_,
            (const float*)bias.weights.d_ptr_,
            M, N
        );
    }
}
void DenseLayer::backward(rocblas_handle blas_handle,
                          hipStream_t stream,
                          const GpuTensor& grad_output,
                          const DenseCache& cache,
                          GpuTensor& grad_input) {
    #define ASSERT_SAME_NUMEL(a,b,msg) \
        if ((a).num_elements_ != (b).num_elements_) \
            throw std::runtime_error("Size mismatch: " msg)

    if (!grad_output.is_allocated() ||
        !cache.input || !cache.input->is_allocated() ||
        !weights.weights.is_allocated()) {
        throw std::runtime_error("Required tensors not allocated for DenseLayer::backward");
    }

    int out_features = weights.weights.dim_size(0);
    int in_features  = weights.weights.dim_size(1);

    // grad_input 텐서가 올바르게 할당되지 않았으면 할당
    if (!grad_input.is_allocated() || grad_input.dims_ != cache.input->dims_) {
        grad_input.allocate(cache.input->dims_);
    }

    // gradient tensor 할당
    weights.allocate_gradients();
    if (has_bias_) {
        bias.allocate_gradients();
    }

    // 배치 크기 재계산
    size_t batch_size_combined = 1;
    for (size_t i = 0; i < grad_output.dims_.size() - 1; ++i) {
        batch_size_combined *= grad_output.dim_size(i);
    }

    // M, N, K를 한 번만 설정하고 이후에 재선언하지 않도록 설정
    int M = static_cast<int>(batch_size_combined);  // 배치 크기
    int K = in_features;                           // 입력 차원
    int N = out_features;                          // 출력 차원

    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));
    const float alpha = 1.0f, beta = 0.0f;
std::cout << "Input dims: " << cache.input->dims_.size() << " dimensions" << std::endl;
std::cout << "Weights dims: " << weights.weights.dims_.size() << " dimensions" << std::endl;
std::cout << "Grad output dims: " << grad_output.dims_.size() << " dimensions" << std::endl;
    // grad_input = grad_output(M×N) * weights(N×K) → (M×K)
    ROCBLAS_CHECK(rocblas_sgemm(
        blas_handle,
        rocblas_operation_none,
        rocblas_operation_none,
        M, K, N,  // M: 배치 크기, N: 출력 차원, K: 입력 차원
        &alpha,
        (const float*)grad_output.d_ptr_, M,  // grad_output 크기 맞춤
        (const float*)weights.weights.d_ptr_, N,  // weights 크기 맞춤
        &beta,
        (float*)grad_input.d_ptr_, M  // grad_input 크기 맞춤
    ));

    // grad_weights = input^T(K×M) * grad_output(M×N) → (K×N) stored row-major

    // 디버깅을 위한 차원 정보 출력
    std::cout << "\n=== DenseLayer Backward 차원 분석 ===" << std::endl;
    std::cout << "cache.input 차원: [";
    for (size_t i = 0; i < cache.input->dims_.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << cache.input->dims_[i];
    }
    std::cout << "] (총 요소: " << cache.input->num_elements_ << ")" << std::endl;
    
    std::cout << "grad_output 차원: [";
    for (size_t i = 0; i < grad_output.dims_.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << grad_output.dims_[i];
    }
    std::cout << "] (총 요소: " << grad_output.num_elements_ << ")" << std::endl;
    
    std::cout << "weights.weights 차원: [";
    for (size_t i = 0; i < weights.weights.dims_.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << weights.weights.dims_[i];
    }
    std::cout << "] (총 요소: " << weights.weights.num_elements_ << ")" << std::endl;

    // grad_weights = input^T(K×M) * grad_output(M×N) → (K×N) stored row-major
    K = cache.input->dim_size(1);  // 입력 차원
    N = grad_output.dim_size(1);   // 출력 차원
    M = grad_output.dim_size(0);   // 배치 크기
    
    std::cout << "계산된 차원: K=" << K << ", N=" << N << ", M=" << M << std::endl;
    std::cout << "=================================" << std::endl;
    // grad_weights 계산
    ROCBLAS_CHECK( rocblas_sgemm(
        blas_handle,
        rocblas_operation_transpose, // Xᵀ : K×M
        rocblas_operation_none,      // dY : M×N
        K, N, M,
        &alpha,
        (const float*)cache.input->d_ptr_,  M,   // lda = M
        (const float*)grad_output.d_ptr_,   M,   // ldb = M
        &beta,
        (float*)weights.grad_weights.d_ptr_, K   // ldc = K (row-major)
    ));

    // grad_bias = reduce_sum(grad_output) over M rows
    if (has_bias_) {
        launch_reduce_sum_kernel(
            stream,
            (float*)bias.grad_weights.d_ptr_,
            (const float*)grad_output.d_ptr_,
            M, N
        );
    }
    printf("[DenseLayer::backward] 역전파 완료 (Grad Input Ptr: %p)\n", grad_input.d_ptr_);
}




void DenseLayer::allocate_gradients() {
    weights.allocate_gradients();
    if (has_bias_) {
        bias.allocate_gradients();
    }
}


// --- LayerNorm Implementation ---LayerNorm::LayerNorm(int hidden_size, float eps, const std::string& name_prefix)
LayerNorm::LayerNorm(int hidden_size, float eps,
                     const std::string& name_prefix)
    : Parameter({hidden_size}, {hidden_size}, name_prefix),   // weights, bias
      eps_(eps)
{
    // γ·β 텐서 메모리 바로 확보
    if (!weights.is_allocated()) weights.allocate({hidden_size});
    if (!bias.is_allocated())    bias.allocate({hidden_size});
}


// [REMOVED] get_parameters() 구현은 헤더에 있으므로 .cpp 파일에서 삭제합니다.

// LayerNorm::forward 구현 (이전 답변의 올바른 버전)
void LayerNorm::forward(hipStream_t stream,
                        const GpuTensor& input,
                        GpuTensor& output,
                        LayerNormCache& cache) {

    // --- ① 파라미터(가중치) 및 출력 텐서 메모리 할당 ---
    int C_cols = input.dims_.back();

    // [FIX] 커널이 사용하기 전에 파라미터가 할당되어 있는지 확인 및 할당
    // 이 파라미터들은 모델 로딩 시점에 미리 할당되는 것이 이상적입니다.
    if (!weights.is_allocated()) {
        weights.allocate({C_cols});
        // weights.load_from_file(...) or other initialization
    }
    if (!bias.is_allocated()) {
        bias.allocate({C_cols});
        // bias.load_from_file(...) or other initialization
    }

    output.allocate(input.dims_);

    // --- ② 역전파를 위한 캐시 메모리 할당 ---
    size_t B_rows = 1;
    for (size_t i = 0; i < input.dims_.size() - 1; ++i) {
        B_rows *= input.dim_size(i);
    }

    cache.input_dims = input.dims_; // 역전파 시 사용하기 위해 입력 차원 저장
    cache.mean.allocate({(int)B_rows});
    cache.rstd.allocate({(int)B_rows});
    // [FIX] 불필요한 allocate_if_smaller 제거

    // --- ③ 커널 실행 ---
    // 모든 메모리 할당이 완료된 후 커널을 실행합니다.
    launch_layer_norm_forward_optimized(
        stream,
        (float*)output.d_ptr_,
        (float*)cache.mean.d_ptr_,
        (float*)cache.rstd.d_ptr_,
        (const float*)input.d_ptr_,
        (const float*)weights.d_ptr_, // 'gamma'
        (const float*)bias.d_ptr_,   // 'beta'
        static_cast<int>(B_rows),
        C_cols,
        1e-5 // Epsilon 값 전달
    );
}

// [FIXED] Parameter 멤버를 올바르게 사용하는 backward 구현
// [FINAL FIXED] LayerNorm::backward 최종 수정 코드
void LayerNorm::backward(hipStream_t stream,
                         const GpuTensor& grad_output,
                         const GpuTensor& original_input,
                         const LayerNormCache& cache,
                         GpuTensor& grad_input) {

    assert(original_input.dims_ == cache.input_dims && "Mismatched dimensions!");

    // 최종 그래디언트 버퍼 확보 ([C] 크기)
    if (!grad_weights.is_allocated()) grad_weights.allocate(weights.dims_);
    if (!grad_bias.is_allocated())    grad_bias.allocate(bias.dims_);

    // 차원 계산
    size_t B_rows = 1;
    for (size_t i = 0; i < cache.input_dims.size() - 1; ++i) {
        B_rows *= cache.input_dims[i];
    }
    int C_cols = cache.input_dims.back();

    // =================================================================
    // [STEP 1] 커널이 사용할 임시 그래디언트 버퍼를 [B, C] 크기로 할당
    // =================================================================
    GpuTensor partial_grad_gamma;
    partial_grad_gamma.allocate({(int)B_rows, C_cols});

    GpuTensor partial_grad_beta;
    partial_grad_beta.allocate({(int)B_rows, C_cols});

    // =================================================================
    // [STEP 2] 커널 호출 시, 이 임시 버퍼를 전달
    // =================================================================
    launch_layer_norm_backward_optimized(
        stream,
        (float*)grad_input.d_ptr_,
        (float*)partial_grad_gamma.d_ptr_, // [FIX] 임시 버퍼 전달
        (float*)partial_grad_beta.d_ptr_,  // [FIX] 임시 버퍼 전달
        (const float*)grad_output.d_ptr_,
        (const float*)original_input.d_ptr_,
        (const float*)weights.d_ptr_,
        (const float*)cache.mean.d_ptr_,
        (const float*)cache.rstd.d_ptr_,
        static_cast<int>(B_rows),
        C_cols,
        eps_
    );


    launch_layer_norm_backward_optimized(
        stream,
        (float*)grad_input.d_ptr_,
        (float*)partial_grad_gamma.d_ptr_,
        (float*)partial_grad_beta.d_ptr_,
        (const float*)grad_output.d_ptr_,
        (const float*)original_input.d_ptr_,
        (const float*)weights.d_ptr_,
        (const float*)cache.mean.d_ptr_,
        (const float*)cache.rstd.d_ptr_,
        static_cast<int>(B_rows),
        C_cols,
        eps_
    );

    // [FIX] 불필요한 zero_out 호출을 삭제합니다.
    // reduce_sum_along_axis_0 커널이 결과값을 바로 덮어쓰기 때문에
    // 미리 0으로 채울 필요가 없습니다.
    reduce_sum_along_axis_0(stream, partial_grad_gamma, grad_weights);
    reduce_sum_along_axis_0(stream, partial_grad_beta,  grad_bias);
}

// [NEW] 누락되었던 allocate_gradients 구현
void LayerNorm::allocate_gradients() {
    Parameter::allocate_gradients();
}
// --- Gelu Implementation ---
void Gelu::forward(hipStream_t stream,
                   const GpuTensor& input,
                   GpuTensor& output,
                   GeluCache& cache) {
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
        input.num_elements_
    );
}

void Gelu::backward(hipStream_t stream,
                    const GpuTensor& grad_output,
                    GpuTensor& grad_input,
                    const GeluCache& cache) {
    if (!grad_output.is_allocated() ||
        !cache.input || !cache.input->is_allocated()) {
        throw std::runtime_error("Required tensors not allocated for Gelu::backward.");
    }
    if (!grad_input.is_allocated() || grad_input.dims_ != grad_output.dims_) {
        grad_input.allocate(grad_output.dims_);
    }
    launch_gelu_backward_kernel(
        stream,
        (float*)grad_input.d_ptr_,
        (const float*)grad_output.d_ptr_,
        (const float*)cache.input->d_ptr_,
        grad_input.num_elements_
    );
}
