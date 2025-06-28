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
    const int seq_len = hidden_states.dims_[1];
    const int hidden_size_val = num_attention_heads_ * attention_head_size_;

    cache.q_proj.allocate({batch_size, seq_len, hidden_size_val});
    cache.k_proj.allocate({batch_size, seq_len, hidden_size_val});
    cache.v_proj.allocate({batch_size, seq_len, hidden_size_val});

    query_.forward(blas_handle, stream, hidden_states, cache.q_proj, cache.q_dense_cache, is_training);
    key_.forward(blas_handle, stream, hidden_states, cache.k_proj, cache.k_dense_cache, is_training);
    value_.forward(blas_handle, stream, hidden_states, cache.v_proj, cache.v_dense_cache, is_training);

    cache.q_reshaped.allocate({batch_size, num_attention_heads_, seq_len, attention_head_size_});
    cache.k_reshaped.allocate({batch_size, num_attention_heads_, seq_len, attention_head_size_});
    cache.v_reshaped.allocate({batch_size, num_attention_heads_, seq_len, attention_head_size_});

    launch_transpose_for_scores_kernel(stream, (float*)cache.q_reshaped.d_ptr_, (const float*)cache.q_proj.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);
    launch_transpose_for_scores_kernel(stream, (float*)cache.k_reshaped.d_ptr_, (const float*)cache.k_proj.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);
    launch_transpose_for_scores_kernel(stream, (float*)cache.v_reshaped.d_ptr_, (const float*)cache.v_proj.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);

    cache.attention_scores.allocate({batch_size, num_attention_heads_, seq_len, seq_len});
    const float alpha_gemm = 1.0f, beta_gemm = 0.0f;
    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));

    long long qk_stride_a = (long long)seq_len * attention_head_size_;
    long long qk_stride_b = (long long)seq_len * attention_head_size_;
    long long score_stride_c = (long long)seq_len * seq_len;
    int batch_count = batch_size * num_attention_heads_;

    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle,
        rocblas_operation_none, rocblas_operation_transpose,
        seq_len, seq_len, attention_head_size_,
        &alpha_gemm,
        (const float*)cache.q_reshaped.d_ptr_, attention_head_size_, qk_stride_a,
        (const float*)cache.k_reshaped.d_ptr_, attention_head_size_, qk_stride_b,
        &beta_gemm,
        (float*)cache.attention_scores.d_ptr_, seq_len, score_stride_c,
        batch_count));

    const float scale_factor = 1.0f / sqrtf(static_cast<float>(attention_head_size_));
    launch_scale_and_mask_kernel(stream, (float*)cache.attention_scores.d_ptr_, attention_mask.is_allocated() ? (const float*)attention_mask.d_ptr_ : nullptr, batch_size, num_attention_heads_, seq_len, seq_len, scale_factor);

    cache.attention_probs.allocate(cache.attention_scores.dims_);
    launch_softmax_kernel(stream, (float*)cache.attention_probs.d_ptr_, (const float*)cache.attention_scores.d_ptr_, batch_size * num_attention_heads_ * seq_len, seq_len);
    
    dropout_.forward(stream, cache.attention_probs, cache.attention_probs_dropout_cache, is_training);

    cache.context_reshaped.allocate({batch_size, num_attention_heads_, seq_len, attention_head_size_});
    long long prob_stride_a = (long long)seq_len * seq_len;
    long long v_stride_b = (long long)seq_len * attention_head_size_;
    long long ctx_stride_c = (long long)seq_len * attention_head_size_;

    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle,
        rocblas_operation_none, rocblas_operation_none,
        attention_head_size_, seq_len, seq_len,
        &alpha_gemm,
        (const float*)cache.v_reshaped.d_ptr_, attention_head_size_, v_stride_b,
        (const float*)cache.attention_probs.d_ptr_, seq_len, prob_stride_a,
        &beta_gemm,
        (float*)cache.context_reshaped.d_ptr_, attention_head_size_, ctx_stride_c,
        batch_count));

    cache.context_layer.allocate({batch_size, seq_len, hidden_size_val});
    launch_transpose_back_kernel(stream, (float*)cache.context_layer.d_ptr_, (const float*)cache.context_reshaped.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);
    cache.input_hidden_states = &hidden_states;
}

void BertSelfAttention::backward(rocblas_handle blas_handle, hipStream_t stream,
                                const GpuTensor& grad_context_layer_output,
                                SelfAttentionCache& cache,
                                GpuTensor& grad_input_hidden_states) {
    if (!grad_context_layer_output.is_allocated() || !cache.input_hidden_states) {
        throw std::runtime_error("Required tensors not allocated for BertSelfAttention backward.");
    }

    const int batch_size = cache.input_hidden_states->dim_size(0);
    const int seq_len = cache.input_hidden_states->dim_size(1);
    const int hidden_size_config = num_attention_heads_ * attention_head_size_;

    if (!grad_input_hidden_states.is_allocated() || grad_input_hidden_states.dims_ != cache.input_hidden_states->dims_) {
        grad_input_hidden_states.allocate(cache.input_hidden_states->dims_);
        grad_input_hidden_states.zero_out(stream);
    }

    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    int batch_count = batch_size * num_attention_heads_;

    GpuTensor grad_context_reshaped;
    grad_context_reshaped.allocate({batch_size, num_attention_heads_, seq_len, attention_head_size_});
    launch_transpose_for_scores_kernel(stream, (float*)grad_context_reshaped.d_ptr_, (const float*)grad_context_layer_output.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);

    GpuTensor grad_attention_probs, grad_v_reshaped;
    grad_attention_probs.allocate(cache.attention_probs.dims_);
    grad_v_reshaped.allocate(cache.v_reshaped.dims_);

    long long grad_ctx_stride = (long long)seq_len * attention_head_size_;
    long long v_reshaped_stride = (long long)seq_len * attention_head_size_;
    long long grad_probs_stride = (long long)seq_len * seq_len;
    long long probs_stride = (long long)seq_len * seq_len;

    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));
    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle,
        rocblas_operation_none, rocblas_operation_transpose,
        seq_len, seq_len, attention_head_size_,
        &alpha,
        (const float*)grad_context_reshaped.d_ptr_, attention_head_size_, grad_ctx_stride,
        (const float*)cache.v_reshaped.d_ptr_, attention_head_size_, v_reshaped_stride,
        &beta_zero,
        (float*)grad_attention_probs.d_ptr_, seq_len, grad_probs_stride,
        batch_count));

    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle,
        rocblas_operation_transpose, rocblas_operation_none,
        attention_head_size_, seq_len, seq_len,
        &alpha,
        (const float*)cache.attention_probs.d_ptr_, seq_len, probs_stride,
        (const float*)grad_context_reshaped.d_ptr_, attention_head_size_, grad_ctx_stride,
        &beta_zero,
        (float*)grad_v_reshaped.d_ptr_, attention_head_size_, v_reshaped_stride,
        batch_count));

    GpuTensor grad_attention_probs_after_dropout;
    grad_attention_probs_after_dropout.allocate(grad_attention_probs.dims_);
    dropout_.backward(stream, grad_attention_probs_after_dropout, cache.attention_probs_dropout_cache);

    GpuTensor grad_scores;
    grad_scores.allocate(cache.attention_scores.dims_);

    int M_softmax = batch_size * num_attention_heads_ * seq_len;
    int N_softmax = seq_len;
    launch_softmax_backward_kernel(stream,
        (float*)grad_scores.d_ptr_,
        (const float*)grad_attention_probs_after_dropout.d_ptr_,
        (const float*)cache.attention_probs.d_ptr_,
        M_softmax, N_softmax);

    const float scale_factor = 1.0f / sqrtf(static_cast<float>(attention_head_size_));
    launch_scale_kernel(stream, (float*)grad_scores.d_ptr_, scale_factor, grad_scores.num_elements_);

    GpuTensor grad_q_reshaped, grad_k_reshaped;
    grad_q_reshaped.allocate(cache.q_reshaped.dims_);
    grad_k_reshaped.allocate(cache.k_reshaped.dims_);

    long long grad_scores_stride = (long long)seq_len * seq_len;
    long long k_reshaped_stride = (long long)seq_len * attention_head_size_;
    long long grad_q_reshaped_stride = (long long)seq_len * attention_head_size_;
    long long q_reshaped_stride = (long long)seq_len * attention_head_size_;
    long long grad_k_reshaped_stride = (long long)seq_len * attention_head_size_;

    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle,
        rocblas_operation_none, rocblas_operation_none,
        attention_head_size_, seq_len, seq_len,
        &alpha,
        (const float*)cache.k_reshaped.d_ptr_, attention_head_size_, k_reshaped_stride,
        (const float*)grad_scores.d_ptr_, seq_len, grad_scores_stride,
        &beta_zero,
        (float*)grad_q_reshaped.d_ptr_, attention_head_size_, grad_q_reshaped_stride,
        batch_count));

    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle,
        rocblas_operation_transpose, rocblas_operation_none,
        attention_head_size_, seq_len, seq_len,
        &alpha,
        (const float*)grad_scores.d_ptr_, seq_len, grad_scores_stride,
        (const float*)cache.q_reshaped.d_ptr_, attention_head_size_, q_reshaped_stride,
        &beta_zero,
        (float*)grad_k_reshaped.d_ptr_, attention_head_size_, grad_k_reshaped_stride,
        batch_count));

    GpuTensor grad_q_proj, grad_k_proj, grad_v_proj;
    grad_q_proj.allocate({batch_size, seq_len, hidden_size_config});
    grad_k_proj.allocate({batch_size, seq_len, hidden_size_config});
    grad_v_proj.allocate({batch_size, seq_len, hidden_size_config});

    launch_transpose_back_kernel(stream, (float*)grad_q_proj.d_ptr_, (const float*)grad_q_reshaped.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);
    launch_transpose_back_kernel(stream, (float*)grad_k_proj.d_ptr_, (const float*)grad_k_reshaped.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);
    launch_transpose_back_kernel(stream, (float*)grad_v_proj.d_ptr_, (const float*)grad_v_reshaped.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);

    value_.backward(blas_handle, stream, grad_v_proj, cache.v_dense_cache, grad_input_hidden_states);
    key_.backward(blas_handle, stream, grad_k_proj, cache.k_dense_cache, grad_input_hidden_states);
    query_.backward(blas_handle, stream, grad_q_proj, cache.q_dense_cache, grad_input_hidden_states);
}

BertAttention::BertAttention(const BertConfig& config, const std::string& name_prefix)
    : self_attention_(config, name_prefix + ".attention.self"),
      output_dense_(config, config.hidden_size, config.hidden_size, name_prefix + ".attention.output.dense"),
      output_dropout_(config.hidden_dropout_prob),
      output_layernorm_(config, name_prefix + ".attention.output.LayerNorm")
      {}

std::vector<Parameter*> BertAttention::get_parameters() {
    auto self_params = self_attention_.get_parameters();
    auto dense_params = output_dense_.get_parameters();
    auto ln_params = output_layernorm_.get_parameters();
    self_params.insert(self_params.end(), dense_params.begin(), dense_params.end());
    self_params.insert(self_params.end(), ln_params.begin(), ln_params.end());
    return self_params;
}

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
    cache.attention_input = &input_tensor;

    self_attention_.forward(blas_handle, stream, input_tensor, attention_mask, cache.self_attention_cache, is_training);

    GpuTensor dense_output;
    dense_output.allocate(cache.self_attention_cache.context_layer.dims_);

    output_dense_.forward(blas_handle, stream, cache.self_attention_cache.context_layer, dense_output, cache.output_dense_cache, is_training);
    output_dropout_.forward(stream, dense_output, cache.output_dropout_cache, is_training);

    GpuTensor attention_residual_sum_output;
    attention_residual_sum_output.allocate(dense_output.dims_);
    launch_elementwise_add_kernel(stream, (float*)attention_residual_sum_output.d_ptr_, (const float*)dense_output.d_ptr_, (const float*)input_tensor.d_ptr_, attention_residual_sum_output.num_elements_);

    output_layernorm_.forward(stream, attention_residual_sum_output, output_tensor, cache.output_layernorm_cache);
}


void BertAttention::backward(rocblas_handle blas_handle, hipStream_t stream,
                            const GpuTensor& grad_final_output,
                            BertAttentionCache& cache,
                            GpuTensor& grad_input_tensor) {
    if (!grad_final_output.is_allocated() || !cache.attention_input) {
        throw std::runtime_error("Required tensors not allocated for BertAttention backward.");
    }
    if(!grad_input_tensor.is_allocated() || grad_input_tensor.dims_ != cache.attention_input->dims_){
        grad_input_tensor.allocate(cache.attention_input->dims_);
        grad_input_tensor.zero_out(stream);
    }

    GpuTensor grad_after_layernorm;
    grad_after_layernorm.allocate(grad_final_output.dims_);
    output_layernorm_.backward(stream, grad_final_output, cache.output_layernorm_cache, grad_after_layernorm);
    
    launch_accumulate_kernel(stream,
                             (float*)grad_input_tensor.d_ptr_,
                             (const float*)grad_after_layernorm.d_ptr_,
                             grad_input_tensor.num_elements_);

    GpuTensor grad_after_dropout;
    grad_after_dropout.allocate(grad_after_layernorm.dims_);
    
    // 수정된 부분: LayerNorm의 역전파 결과를 Dropout의 역전파 입력으로 복사합니다.
    grad_after_dropout.copy_from_gpu(grad_after_layernorm, stream);

    // 이제 올바른 Gradient가 전달됩니다.
    output_dropout_.backward(stream, grad_after_dropout, cache.output_dropout_cache);

    GpuTensor grad_after_dense;
    grad_after_dense.allocate(cache.self_attention_cache.context_layer.dims_); 
    output_dense_.backward(blas_handle, stream, grad_after_dropout, cache.output_dense_cache, grad_after_dense);

    self_attention_.backward(blas_handle, stream, grad_after_dense, cache.self_attention_cache, grad_input_tensor);
}