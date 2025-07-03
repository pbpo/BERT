#include "attention_hip.hpp"
#include "common_hip.hpp"
#include "hip_kernels.hpp"

BertSelfAttention::BertSelfAttention(const BertConfig& config, const std::string& name_prefix)
    : query_(config, config.hidden_size, config.hidden_size, name_prefix + ".query"),
      key_(config, config.hidden_size, config.hidden_size, name_prefix + ".key"),
      value_(config, config.hidden_size, config.hidden_size, name_prefix + ".value"),
      dropout_(config.attention_probs_dropout_prob),
      num_attention_heads_(config.num_attention_heads),
      attention_head_size_(config.hidden_size / config.num_attention_heads) {
    if (config.hidden_size % config.num_attention_heads != 0) {
        throw std::runtime_error("Hidden size must be divisible by the number of attention heads");
    }
}

std::vector<Parameter*> BertSelfAttention::get_parameters() {
    std::vector<Parameter*> params;
    auto q_params = query_.get_parameters();
    auto k_params = key_.get_parameters();
    auto v_params = value_.get_parameters();
    params.insert(params.end(), q_params.begin(), q_params.end());
    params.insert(params.end(), k_params.begin(), k_params.end());
    params.insert(params.end(), v_params.begin(), v_params.end());
    return params;
}

void BertSelfAttention::forward(rocblas_handle blas_handle, hipStream_t stream,
                                const GpuTensor& hidden_states, const GpuTensor& attention_mask,
                                SelfAttentionCache& cache, bool is_training) {
    const int batch_size = hidden_states.dims_[0];
    const int seq_len    = hidden_states.dims_[1];
    const int head_dim   = attention_head_size_;
    const int hidden_val = num_attention_heads_ * head_dim;
    int batch_count = batch_size * num_attention_heads_;
    const float alpha = 1.0f, beta = 0.0f;

    // 1) Q/K/V projection
    cache.q_proj.allocate({batch_size, seq_len, hidden_val});
    cache.k_proj.allocate({batch_size, seq_len, hidden_val});
    cache.v_proj.allocate({batch_size, seq_len, hidden_val});
    query_.forward(blas_handle, stream, hidden_states, cache.q_proj, cache.q_dense_cache, is_training);
    key_.forward  (blas_handle, stream, hidden_states, cache.k_proj, cache.k_dense_cache, is_training);
    value_.forward(blas_handle, stream, hidden_states, cache.v_proj, cache.v_dense_cache, is_training);

    // 2) reshape for batched GEMM: [B, H, S, D]
cache.q_reshaped.allocate({batch_size, num_attention_heads_, seq_len, head_dim});
cache.k_reshaped.allocate({batch_size, num_attention_heads_, seq_len, head_dim});
cache.v_reshaped.allocate({batch_size, num_attention_heads_, seq_len, head_dim});
launch_transpose_for_scores_kernel(stream,
    (float*)cache.q_reshaped.d_ptr_, (const float*)cache.q_proj.d_ptr_,
    batch_size, seq_len, num_attention_heads_, head_dim);
launch_transpose_for_scores_kernel(stream,
    (float*)cache.k_reshaped.d_ptr_, (const float*)cache.k_proj.d_ptr_,
    batch_size, seq_len, num_attention_heads_, head_dim);
launch_transpose_for_scores_kernel(stream,
    (float*)cache.v_reshaped.d_ptr_, (const float*)cache.v_proj.d_ptr_,
    batch_size, seq_len, num_attention_heads_, head_dim);

// 3) Q·Kᵀ → attention_scores
cache.attention_scores.allocate({batch_size, num_attention_heads_, seq_len, seq_len});
{
    const int M = seq_len;          // A 행(row) 수 = C 행(row) 수
    const int N = seq_len;          // Bᵀ 열(col) 수 = C 열(col) 수
    const int K = head_dim;         // A 열(col) 수 = Bᵀ 행(row) 수

    // Stride: 각 배치별로 연속된 행렬 간 떨어진 요소 수
    const long long strideA = (long long)M * K;  // A: M×K
    const long long strideB = (long long)N * K;  // B: N×K
    const long long strideC = (long long)M * N;  // C: M×N

    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));
    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(
        blas_handle,
        rocblas_operation_none,      // A 그대로: M×K
        rocblas_operation_transpose, // Bᵀ: (K×N)
        M, N, K,
        &alpha,
        /* A */ (const float*)cache.q_reshaped.d_ptr_, /* lda = M */ M, strideA,
        /* B */ (const float*)cache.k_reshaped.d_ptr_, /* ldb = N */ N, strideB,
        &beta,
        /* C */ (float*)cache.attention_scores.d_ptr_, /* ldc = M */ M, strideC,
        batch_count));
}
    // 4) scale + mask + softmax + dropout
    float scale = 1.0f / std::sqrt((float)head_dim);
    launch_scale_and_mask_kernel(stream,
        (float*)cache.attention_scores.d_ptr_,
        attention_mask.is_allocated() ? (const float*)attention_mask.d_ptr_ : nullptr,
        batch_size, num_attention_heads_, seq_len, seq_len, scale);
    cache.attention_probs.allocate(cache.attention_scores.dims_);
    launch_softmax_kernel(stream,
        (float*)cache.attention_probs.d_ptr_,
        (const float*)cache.attention_scores.d_ptr_,
        batch_size * num_attention_heads_ * seq_len, seq_len);
    dropout_.forward(stream, cache.attention_probs, cache.attention_probs_dropout_cache, is_training);

    // 5) attention_probs · V → context_reshaped
    cache.context_reshaped.allocate({batch_size, num_attention_heads_, seq_len, head_dim});
    {
        int M = seq_len, N = head_dim, K = seq_len;
        long long strideA = (long long)seq_len * seq_len;
        long long strideB = (long long)seq_len * head_dim;
        long long strideC = (long long)seq_len * head_dim;
        ROCBLAS_CHECK(rocblas_sgemm_strided_batched(
            blas_handle,
            rocblas_operation_none,     // A: seq_len×seq_len
            rocblas_operation_none,     // B: seq_len×head_dim
            M, N, K,
            &alpha,
            (const float*)cache.attention_scores.d_ptr_, M, strideA,
            (const float*)cache.v_reshaped.d_ptr_,       K, strideB,
            &beta,
            (float*)cache.context_reshaped.d_ptr_,       M, strideC,
            batch_count));
    }

    // 6) transpose back → context_layer [B, S, H*D]
    cache.context_layer.allocate({batch_size, seq_len, hidden_val});
    launch_transpose_back_kernel(stream,
        (float*)cache.context_layer.d_ptr_,
        (const float*)cache.context_reshaped.d_ptr_,
        batch_size, seq_len, num_attention_heads_, head_dim);

   
}

// attention_hip.cpp

// attention_hip.cpp

void BertSelfAttention::backward(rocblas_handle blas_handle, hipStream_t stream,
                                const GpuTensor& grad_context_layer_output,
                                SelfAttentionCache& cache,
                                GpuTensor& grad_input_hidden_states) {
    if (!grad_context_layer_output.is_allocated()) {
        throw std::runtime_error("Input gradient tensor not allocated for BertSelfAttention backward.");
    }
  printf("[DEBUG] BertSelfAttention::backward - num_attention_heads: %d, attention_head_size: %d\n",
           num_attention_heads_, attention_head_size_);
    const int batch_size = cache.q_proj.dim_size(0);
    const int seq_len = cache.q_proj.dim_size(1);
    const int hidden_size_config = num_attention_heads_ * attention_head_size_;

    grad_input_hidden_states.zero_out(stream);

    // --- 캐시로부터 임시 텐서 할당 ---
    cache.grad_context_reshaped.allocate({batch_size, num_attention_heads_, seq_len, attention_head_size_});
    cache.grad_attention_probs.allocate(cache.attention_probs.dims_);
    cache.grad_v_reshaped.allocate(cache.v_reshaped.dims_);
    cache.grad_scores.allocate(cache.attention_scores.dims_);
    cache.grad_q_reshaped.allocate(cache.q_reshaped.dims_);
    cache.grad_k_reshaped.allocate(cache.k_reshaped.dims_);
    cache.grad_q_proj.allocate({batch_size, seq_len, hidden_size_config});
    cache.grad_k_proj.allocate({batch_size, seq_len, hidden_size_config});
    cache.grad_v_proj.allocate({batch_size, seq_len, hidden_size_config});
    cache.grad_temp_for_accumulation.allocate(grad_input_hidden_states.dims_);

    // --- 역전파 계산 ---
    const float alpha = 1.0f, beta_zero = 0.0f;
    int batch_count = batch_size * num_attention_heads_;
    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));

    launch_transpose_for_scores_kernel(stream, (float*)cache.grad_context_reshaped.d_ptr_, (const float*)grad_context_layer_output.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);

    // [수정] 모든 sgemm 호출의 lda, ldb, ldc 값을 행렬의 행(row) 수에 맞게 수정합니다.

    // 1. grad_attention_probs = grad_context_reshaped * v_reshaped^T
    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle, rocblas_operation_none, rocblas_operation_transpose,
        /* M,N,K */ seq_len, seq_len, attention_head_size_, &alpha,
        /* A */ (const float*)cache.grad_context_reshaped.d_ptr_, /* lda */ seq_len, (long long)seq_len * attention_head_size_,
        /* B */ (const float*)cache.v_reshaped.d_ptr_,             /* ldb */ seq_len, (long long)seq_len * attention_head_size_,
        &beta_zero,
        /* C */ (float*)cache.grad_attention_probs.d_ptr_,        /* ldc */ seq_len, (long long)seq_len * seq_len,
        batch_count));

    // 2. grad_v_reshaped = attention_probs^T * grad_context_reshaped
    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle, rocblas_operation_transpose, rocblas_operation_none,
        /* M,N,K */ attention_head_size_, seq_len, seq_len, &alpha,
        /* A */ (const float*)cache.attention_probs.d_ptr_,      /* lda */ seq_len, (long long)seq_len * seq_len,
        /* B */ (const float*)cache.grad_context_reshaped.d_ptr_, /* ldb */ seq_len, (long long)seq_len * attention_head_size_,
        &beta_zero,
        /* C */ (float*)cache.grad_v_reshaped.d_ptr_,             /* ldc */ attention_head_size_, (long long)attention_head_size_ * seq_len,
        batch_count));

    dropout_.backward(stream, cache.grad_attention_probs, cache.grad_attention_probs, cache.attention_probs_dropout_cache);

    launch_softmax_backward_kernel(stream, (float*)cache.grad_scores.d_ptr_, (const float*)cache.grad_attention_probs.d_ptr_, (const float*)cache.attention_probs.d_ptr_, batch_size * num_attention_heads_ * seq_len, seq_len);
    
    const float scale_factor = 1.0f / sqrtf(static_cast<float>(attention_head_size_));
    launch_scale_kernel(stream, (float*)cache.grad_scores.d_ptr_, scale_factor, cache.grad_scores.num_elements_);
    
    // 3. grad_q_reshaped = grad_scores * k_reshaped
    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle, rocblas_operation_none, rocblas_operation_none,
        /* M,N,K */ seq_len, attention_head_size_, seq_len, &alpha,
        /* A */ (const float*)cache.grad_scores.d_ptr_,  /* lda */ seq_len, (long long)seq_len * seq_len,
        /* B */ (const float*)cache.k_reshaped.d_ptr_,    /* ldb */ seq_len, (long long)seq_len * attention_head_size_,
        &beta_zero,
        /* C */ (float*)cache.grad_q_reshaped.d_ptr_,    /* ldc */ seq_len, (long long)seq_len * attention_head_size_,
        batch_count));

    // 4. grad_k_reshaped = grad_scores^T * q_reshaped
    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle, rocblas_operation_transpose, rocblas_operation_none,
        /* M,N,K */ seq_len, attention_head_size_, seq_len, &alpha,
        /* A */ (const float*)cache.grad_scores.d_ptr_,  /* lda */ seq_len, (long long)seq_len * seq_len,
        /* B */ (const float*)cache.q_reshaped.d_ptr_,    /* ldb */ seq_len, (long long)seq_len * attention_head_size_,
        &beta_zero,
        /* C */ (float*)cache.grad_k_reshaped.d_ptr_,    /* ldc */ seq_len, (long long)seq_len * attention_head_size_,
        batch_count));

    launch_transpose_back_kernel(stream, (float*)cache.grad_q_proj.d_ptr_, (const float*)cache.grad_q_reshaped.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);
    launch_transpose_back_kernel(stream, (float*)cache.grad_k_proj.d_ptr_, (const float*)cache.grad_k_reshaped.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);
    launch_transpose_back_kernel(stream, (float*)cache.grad_v_proj.d_ptr_, (const float*)cache.grad_v_reshaped.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);

    // 그래디언트 누적
    value_.backward(blas_handle, stream, cache.grad_v_proj, cache.v_dense_cache, grad_input_hidden_states);

    cache.grad_temp_for_accumulation.zero_out(stream);
    key_.backward(blas_handle, stream, cache.grad_k_proj, cache.k_dense_cache, cache.grad_temp_for_accumulation);
    launch_accumulate_kernel(stream, (float*)grad_input_hidden_states.d_ptr_, (const float*)cache.grad_temp_for_accumulation.d_ptr_, grad_input_hidden_states.num_elements_);

    cache.grad_temp_for_accumulation.zero_out(stream);
    query_.backward(blas_handle, stream, cache.grad_q_proj, cache.q_dense_cache, cache.grad_temp_for_accumulation);
    launch_accumulate_kernel(stream, (float*)grad_input_hidden_states.d_ptr_, (const float*)cache.grad_temp_for_accumulation.d_ptr_, grad_input_hidden_states.num_elements_);
}

BertAttention::BertAttention(const BertConfig& config, const std::string& name_prefix)
    : self_attention_(config, name_prefix + ".attention.self"),
      output_dense_(config, config.hidden_size, config.hidden_size, name_prefix + ".attention.output.dense"),
      output_dropout_(config.hidden_dropout_prob),
      // [FIXED] LayerNorm 생성자에 config의 멤버 변수를 직접 전달
      output_layernorm_(config.hidden_size, config.layer_norm_eps, name_prefix + ".attention.output.LayerNorm")
      {}

std::vector<Parameter*> BertAttention::get_parameters() {
    auto self_params = self_attention_.get_parameters();
    auto dense_params = output_dense_.get_parameters();
    // [FIXED] 이제 output_layernorm_에 get_parameters()가 있으므로 정상 동작
    auto ln_params = output_layernorm_.get_parameters();
    
    self_params.insert(self_params.end(), dense_params.begin(), dense_params.end());
    self_params.insert(self_params.end(), ln_params.begin(), ln_params.end());
    return self_params;
}
// attention_hip.hpp 또는 attention_hip.cpp

// attention_hip.cpp
void BertAttention::forward(rocblas_handle blas_handle, hipStream_t stream,
                           const GpuTensor& input_tensor,
                           const GpuTensor& attention_mask,
                           GpuTensor& output_tensor,
                           BertAttentionCache& cache,
                           bool is_training) {
    if (!input_tensor.is_allocated()) {
        throw std::runtime_error("Input tensor not allocated for BertAttention forward.");
    }
    if (!output_tensor.is_allocated() || output_tensor.dims_ != input_tensor.dims_) {
        output_tensor.allocate(input_tensor.dims_);
    }

    // ======================= [핵심 수정 1] =======================
    // backward에서 안전하게 사용하기 위해, 입력 데이터를 캐시의 내부 저장소로 복사합니다.
    cache.attention_input.copy_from_gpu(input_tensor, stream);
    // ==========================================================

    // 1. Self-Attention
    // [핵심 수정 2] 원본 input_tensor 대신 캐시에 복사된 안전한 버전을 SelfAttention에 전달합니다.
    self_attention_.forward(blas_handle, stream, cache.attention_input, attention_mask, cache.self_attention_cache, is_training);

    // 2. Self-Attention의 출력을 받아 Dense Layer와 Dropout을 통과시킵니다.
    // Self-Attention의 출력은 cache.self_attention_cache.context_layer에 저장되어 있습니다.
    GpuTensor dense_output;
    dense_output.allocate(cache.self_attention_cache.context_layer.dims_);
    output_dense_.forward(blas_handle, stream, cache.self_attention_cache.context_layer, dense_output, cache.output_dense_cache, is_training);
    output_dropout_.forward(stream, dense_output, cache.output_dropout_cache, is_training);

    // 3. Residual Connection (잔차 연결)
    // Dropout 결과와 원본 입력(의 안전한 복사본)을 더합니다.
    GpuTensor attention_residual_sum_output;
    attention_residual_sum_output.allocate(dense_output.dims_);
    launch_elementwise_add_kernel(stream,
                                  (float*)attention_residual_sum_output.d_ptr_,
                                  (const float*)dense_output.d_ptr_,
                                  (const float*)cache.attention_input.d_ptr_, // [핵심 수정 3] 캐시에 저장된 안전한 복사본 사용
                                  attention_residual_sum_output.num_elements_);

    // 4. 최종 LayerNorm
    // backward를 위해 계산 결과를 캐시에 복사하여 저장합니다.
    cache.cached_residual_sum_output.copy_from_gpu(attention_residual_sum_output, stream);
    output_layernorm_.forward(stream, attention_residual_sum_output, output_tensor, cache.output_layernorm_cache);
}

void BertAttention::backward(rocblas_handle blas_handle, hipStream_t stream,
                            const GpuTensor& grad_final_output,
                            BertAttentionCache& cache,
                            GpuTensor& grad_input_tensor) {
    printf("[BertAttention::backward] 시작 (Grad Output Ptr: %p)\n", grad_final_output.d_ptr_);

    if (!grad_final_output.is_allocated() || !cache.attention_input.is_allocated()) {
        throw std::runtime_error("Required tensors not allocated for BertAttention backward.");
    }

    if (!grad_input_tensor.is_allocated() || grad_input_tensor.dims_ != cache.attention_input.dims_) {
        grad_input_tensor.allocate(cache.attention_input.dims_);
    } else {
        grad_input_tensor.zero_out(stream);
    }
    
    // --- 임시 텐서들 (이 함수 내에서만 사용) ---
    GpuTensor grad_after_layernorm;
    GpuTensor grad_after_dropout;
    GpuTensor grad_after_dense;

    // ======================= [디버깅 코드 블록] =======================
    // 헬퍼 람다 함수 (GPU 텐서의 앞 5개 원소를 출력)
    auto print_tensor_head = [&](const std::string& name, const GpuTensor& tensor) {
        if (!tensor.is_allocated()) {
            printf("  [DEBUG] %s: Not Allocated\n", name.c_str());
            return;
        }
        printf("  [DEBUG] %s (ptr: %p)\n", name.c_str(), tensor.d_ptr_);
        std::vector<float> host_data(10); // Print first 10
        hipError_t err = hipMemcpy(host_data.data(), tensor.d_ptr_, std::min((size_t)10, tensor.num_elements_) * sizeof(float), hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            printf("  [ERROR] %s: 메모리 읽기 실패: %s\n", name.c_str(), hipGetErrorString(err));
            return;
        }
        for (int i = 0; i < 5; ++i) {
           printf("    > %s[%d] = %.32f\n", name.c_str(), i, host_data[i]);
        }
    };
    printf("\n--- BertAttention 역전파 단계별 값 추적 ---\n");
    // =================================================================

    // 1. LayerNorm 역전파
     grad_after_layernorm.allocate(grad_final_output.dims_);
    output_layernorm_.backward(
        stream,
        grad_final_output,
        // 👇 존재하지 않던 변수 대신, cache에 저장된 텐서를 사용합니다.
        cache.cached_residual_sum_output,
        cache.output_layernorm_cache,
        grad_after_layernorm
    );


    HIP_CHECK(hipStreamSynchronize(stream));
    printf("--- 1. LayerNorm 역전파 직후 ---\n");
    print_tensor_head("grad_after_layernorm", grad_after_layernorm);

    // 2. Dropout 역전파
    grad_after_dropout.allocate(grad_after_layernorm.dims_);
    grad_after_dropout.copy_from_gpu(grad_after_layernorm, stream);
    output_dropout_.backward(stream, grad_after_dropout, grad_after_dropout, cache.output_dropout_cache);

    HIP_CHECK(hipStreamSynchronize(stream));
    printf("--- 2. Dropout 역전파 직후 ---\n");
    print_tensor_head("grad_after_dropout", grad_after_dropout);

    // 3. Dense Layer 역전파
    grad_after_dense.allocate(cache.self_attention_cache.context_layer.dims_); 
    output_dense_.backward(blas_handle, stream, grad_after_dropout, cache.output_dense_cache, grad_after_dense);

    HIP_CHECK(hipStreamSynchronize(stream));
    printf("--- 3. Output Dense 역전파 직후 ---\n");
    print_tensor_head("grad_after_dense (Before Accum)", grad_after_dense);

    // 4. 잔차 연결의 그래디언트 합산
    launch_accumulate_kernel(stream,
                             (float*)grad_after_dense.d_ptr_,
                             (const float*)grad_after_layernorm.d_ptr_,
                             grad_after_dense.num_elements_);

    HIP_CHECK(hipStreamSynchronize(stream));
    printf("--- 4. 잔차 연결 그래디언트 합산 직후 ---\n");
    print_tensor_head("grad_after_dense (After Accum)", grad_after_dense);

    // 5. Self-Attention 역전파
    printf("--- 5. Self-Attention 역전파 시작 ---\n");
    self_attention_.backward(blas_handle, stream, grad_after_dense, cache.self_attention_cache, grad_input_tensor);

    HIP_CHECK(hipStreamSynchronize(stream));
    printf("--- 6. Self-Attention 역전파 완료 ---\n");
    print_tensor_head("grad_input_tensor (Final)", grad_input_tensor);
    printf("---------------------------------------\n\n");
}