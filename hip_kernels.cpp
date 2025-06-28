#include "hip_kernels.hpp"
#include "common_hip.hpp" // For HIP_CHECK, GpuTensor, THREADS_PER_BLOCK_DEFAULT
#include <algorithm> // For std::min

// ============================================================================
// Kernel Launcher Implementations
// ============================================================================

// --- Batched GEMM Pointer Setup ---
void launch_setup_batched_gemm_pointers(
    hipStream_t stream,
    GpuPtrArray& A_ptrs, GpuPtrArray& B_ptrs, GpuPtrArray& C_ptrs,
    const GpuTensor& A_tensor, const GpuTensor& B_tensor, GpuTensor& C_tensor,
    size_t batch_count)
{
    if (batch_count == 0) return;

    // 배치 내 각 행렬의 시작 주소 사이의 거리(element 수)를 계산합니다.
    size_t stride_A = (A_tensor.num_elements_ > 0) ? A_tensor.num_elements_ / batch_count : 0;
    size_t stride_B = (B_tensor.num_elements_ > 0) ? B_tensor.num_elements_ / batch_count : 0;
    size_t stride_C = (C_tensor.num_elements_ > 0) ? C_tensor.num_elements_ / batch_count : 0;

    dim3 block_dim(THREADS_PER_BLOCK_DEFAULT);
    dim3 grid_dim((batch_count + block_dim.x - 1) / block_dim.x);

    hipLaunchKernelGGL(
        setup_batched_gemm_pointers_kernel,
        grid_dim, block_dim, 0, stream,
        A_ptrs.d_ptr_, B_ptrs.d_ptr_, C_ptrs.d_ptr_,
        A_tensor.d_ptr_(), B_tensor.d_ptr_(), C_tensor.d_ptr_(),
        batch_count, stride_A, stride_B, stride_C
    );
    HIP_CHECK(hipGetLastError());
}

// --- Dropout ---
void launch_dropout_forward(
    hipStream_t stream, float* output, float* mask, const float* input,
    size_t num_elements, float prob, float scale, unsigned long long seed)
{
    if (num_elements == 0) return;
    dim3 grid_dim((num_elements + THREADS_PER_BLOCK_DEFAULT - 1) / THREADS_PER_BLOCK_DEFAULT);
    dim3 block_dim(THREADS_PER_BLOCK_DEFAULT);

    hipLaunchKernelGGL(dropout_forward_kernel_impl, grid_dim, block_dim, 0, stream,
                       output, input, mask, num_elements, prob, scale, seed, nullptr);
    HIP_CHECK(hipGetLastError());
}

void launch_dropout_backward(
    hipStream_t stream, float* grad_input, const float* grad_output, const float* mask,
    size_t num_elements, float scale)
{
    if (num_elements == 0) return;
    dim3 grid_dim((num_elements + THREADS_PER_BLOCK_DEFAULT - 1) / THREADS_PER_BLOCK_DEFAULT);
    dim3 block_dim(THREADS_PER_BLOCK_DEFAULT);

    hipLaunchKernelGGL(dropout_backward_kernel_impl, grid_dim, block_dim, 0, stream,
                       grad_input, grad_output, mask, num_elements, scale);
    HIP_CHECK(hipGetLastError());
}

// --- Layer Normalization (Optimized) ---
void launch_layer_norm_forward_optimized(
    hipStream_t stream, float* out, float* mean, float* rstd,
    const float* inp, const float* gamma, const float* beta,
    int B, int C, float epsilon)
{
    if (B == 0 || C == 0) return;
    // 각 배치를 하나의 블록이 처리하도록 grid 설정
    dim3 grid(B); 
    // 블록 내 스레드들은 C 차원에 대해 병렬 처리
    dim3 block(std::min(C, (int)THREADS_PER_BLOCK_DEFAULT)); 

    // Warp रिडक्शन과 파라미터 브로드캐스팅을 위한 공유 메모리 크기 계산
    size_t shared_mem_size = ((block.x + warpSize - 1) / warpSize) * sizeof(float) + 2 * sizeof(float);

    hipLaunchKernelGGL(layer_norm_forward_kernel_warp_optimized_impl, grid, block, shared_mem_size, stream,
                       out, mean, rstd, inp, gamma, beta, B, C, epsilon);
    HIP_CHECK(hipGetLastError());
}

void launch_layer_norm_backward_optimized(
    hipStream_t stream, float* grad_input, float* grad_gamma, float* grad_beta,
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* rstd,
    int B, int C)
{
    if (B == 0 || C == 0) return;
    dim3 grid(B);
    dim3 block(std::min(C, (int)THREADS_PER_BLOCK_DEFAULT));
    size_t shared_mem_size = ((block.x + warpSize - 1) / warpSize) * sizeof(float) + 2 * sizeof(float);

    hipLaunchKernelGGL(layer_norm_backward_kernel_optimized_impl, grid, block, shared_mem_size, stream,
                       grad_input, grad_gamma, grad_beta,
                       grad_output, input, gamma, mean, rstd,
                       B, C);
    HIP_CHECK(hipGetLastError());
}

// --- GELU and Bias Kernels ---
void launch_add_bias_gelu_kernel(
    hipStream_t stream, float* output, const float* input, const float* bias, int M, int N)
{
    if (M == 0 || N == 0) return;
    // 2D 데이터(M, N) 처리를 위한 2D 스레드 블록 설정
    dim3 threads(16, 16); 
    // 커널이 (col, row) 순서로 인덱싱하므로 grid도 (N, M)에 맞춰 설정
    dim3 grid((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    hipLaunchKernelGGL(add_bias_gelu_kernel_impl, grid, threads, 0, stream, output, input, bias, M, N);
    HIP_CHECK(hipGetLastError());
}

void launch_gelu_add_bias_backward_kernel(
    hipStream_t stream,
    float* grad_input_before_bias,
    float* grad_bias,
    const float* grad_output_after_gelu,
    const float* input_before_gelu,
    int M,
    int N)
{
    if (M == 0 || N == 0) return;

    dim3 threads(16, 16);
    dim3 grid((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    hipLaunchKernelGGL(gelu_add_bias_backward_kernel_impl, grid, threads, 0, stream,
                       grad_input_before_bias, grad_bias,
                       grad_output_after_gelu, input_before_gelu,
                       M, N);
    HIP_CHECK(hipGetLastError());
}

void launch_gelu_backward_kernel(
    hipStream_t stream, float* grad_input, const float* grad_output, const float* input, size_t num_elements)
{
    if (num_elements == 0) return;
    dim3 block_dim(THREADS_PER_BLOCK_DEFAULT);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x);
    hipLaunchKernelGGL(gelu_backward_kernel_impl, grid_dim, block_dim, 0, stream, grad_input, grad_output, input, num_elements);
    HIP_CHECK(hipGetLastError());
}

void launch_gelu_forward_kernel(
    hipStream_t stream, float* output, const float* input, size_t num_elements)
{
    if (num_elements == 0) return;
    dim3 block_dim(THREADS_PER_BLOCK_DEFAULT);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x);
    hipLaunchKernelGGL(gelu_forward_kernel_impl, grid_dim, block_dim, 0, stream, output, input, num_elements);
    HIP_CHECK(hipGetLastError());
}

// --- Reduction Kernels ---
void launch_reduce_sum_axis0_add_kernel(
    hipStream_t stream, float* out_grad, const float* in_grad, int M, int N)
{
    if (M == 0 || N == 0) return;
    // N개의 feature 각각에 대해 블록 할당
    dim3 grid_dim(N); 
    // 블록 내 스레드들은 M 차원에 대해 합산 수행
    dim3 block_dim(std::min(M, (int)THREADS_PER_BLOCK_DEFAULT)); 
    size_t shared_mem_size = block_dim.x * sizeof(float);

    hipLaunchKernelGGL(reduce_sum_axis0_add_kernel_impl, grid_dim, block_dim, shared_mem_size, stream,
                       out_grad, in_grad, M, N);
    HIP_CHECK(hipGetLastError());
}

void launch_reduce_sum_axis1_add_kernel(
    hipStream_t stream, float* out_grad, const float* in_grad, int rows, int cols)
{
    if (rows == 0 || cols == 0) return;
    // `rows` 개수만큼 블록 할당 (각 블록이 한 행 처리)
    dim3 grid_dim(rows);
    // 블록 내 스레드들은 `cols` 차원에 대해 합산 수행
    dim3 block_dim(std::min(cols, (int)THREADS_PER_BLOCK_DEFAULT));
    size_t shared_mem_size = block_dim.x * sizeof(float);
    hipLaunchKernelGGL(reduce_sum_axis1_add_kernel_impl, grid_dim, block_dim, shared_mem_size, stream,
                       out_grad, in_grad, rows, cols);
    HIP_CHECK(hipGetLastError());
}

// --- Softmax & Loss Kernels ---
void launch_softmax_cross_entropy_loss_backward_optimized(
    hipStream_t stream, float* grad_logits, const float* logits,
    const int* labels, float* total_loss,
    int B, int S, int V)
{
    if (B == 0 || S == 0 || V == 0) return;
    // 각 토큰(B, S)에 대해 하나의 블록 할당
    dim3 grid(B, S); 
    // 블록 내 스레드들은 Vocab(V) 차원에 대해 병렬 처리
    dim3 block(std::min(V, (int)THREADS_PER_BLOCK_DEFAULT));
    size_t shared_mem_size = ((block.x + warpSize - 1) / warpSize) * sizeof(float) + 2 * sizeof(float);

    hipLaunchKernelGGL(softmax_cross_entropy_loss_backward_kernel_optimized_impl,
                       grid, block, shared_mem_size, stream,
                       grad_logits, logits, labels, total_loss, B, S, V);
    HIP_CHECK(hipGetLastError());
}

// --- AdamW Optimizer ---
void launch_adamw_update_kernel(
    hipStream_t stream, float* params, const float* grads, float* m, float* v,
    float lr, float beta1, float beta2, float eps, float weight_decay, int t, size_t num_elements)
{
    if (num_elements == 0) return;
    dim3 block_dim(THREADS_PER_BLOCK_DEFAULT);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x);
    hipLaunchKernelGGL(adamw_update_kernel_impl, grid_dim, block_dim, 0, stream,
                       params, grads, m, v, lr, beta1, beta2, eps, weight_decay, t, num_elements);
    HIP_CHECK(hipGetLastError());
}

// --- Other Utility Kernels ---
void launch_add_bias_kernel(
    hipStream_t stream, float* output, const float* input, const float* bias, int M, int N)
{
    if (M == 0 || N == 0) return;
    dim3 threads(16, 16);
    dim3 grid((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    hipLaunchKernelGGL(add_bias_kernel_impl, grid, threads, 0, stream, output, input, bias, M, N);
    HIP_CHECK(hipGetLastError());
}

void launch_scale_kernel(
    hipStream_t stream, float* data, float scale, size_t num_elements)
{
    if (num_elements == 0) return;
    dim3 block_dim(THREADS_PER_BLOCK_DEFAULT);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x);
    hipLaunchKernelGGL(scale_kernel_impl, grid_dim, block_dim, 0, stream, data, scale, num_elements);
    HIP_CHECK(hipGetLastError());
}

// --- Embedding Kernels ---
void launch_add_embeddings_kernel(
    hipStream_t stream, float* output, const int* input_ids, const int* token_type_ids,
    const float* word_embeddings, const float* position_embeddings,
    const float* token_type_embeddings, int batch_size, int seq_len,
    int hidden_size, int vocab_size, int max_position_embeddings)
{
    if (batch_size == 0 || seq_len == 0 || hidden_size == 0) return;
    // (batch, seq) 각 아이템에 대해 블록 할당
    dim3 grid(batch_size, seq_len);
    // 블록 내 스레드들은 hidden_size에 대해 병렬 처리
    dim3 block(std::min(hidden_size, (int)THREADS_PER_BLOCK_DEFAULT)); 
    hipLaunchKernelGGL(add_embeddings_kernel_impl, grid, block, 0, stream,
                       output, input_ids, token_type_ids, word_embeddings, position_embeddings,
                       token_type_embeddings, batch_size, seq_len, hidden_size, vocab_size, max_position_embeddings);
    HIP_CHECK(hipGetLastError());
}

void launch_embedding_backward_kernel(
    hipStream_t stream, float* grad_word_embeddings, float* grad_position_embeddings,
    float* grad_token_type_embeddings, const float* grad_output,
    const int* input_ids, const int* token_type_ids,
    int batch_size, int seq_len, int hidden_size)
{
    if (batch_size == 0 || seq_len == 0 || hidden_size == 0) return;
    dim3 grid(batch_size, seq_len);
    dim3 block(std::min(hidden_size, (int)THREADS_PER_BLOCK_DEFAULT));
    hipLaunchKernelGGL(embedding_backward_kernel_impl, grid, block, 0, stream,
                       grad_word_embeddings, grad_position_embeddings, grad_token_type_embeddings,
                       grad_output, input_ids, token_type_ids, batch_size, seq_len, hidden_size);
    HIP_CHECK(hipGetLastError());
}

// --- Transpose Kernels for Attention ---
void launch_transpose_for_scores_kernel(
    hipStream_t stream, float* output, const float* input,
    int batch_size, int seq_len, int num_heads, int head_size)
{
    if (batch_size * seq_len * num_heads * head_size == 0) return;
    int hidden_size = num_heads * head_size;
    // (batch, seq) 각 아이템에 대해 블록 할당
    dim3 grid(batch_size, seq_len);
    // 블록 내 스레드들이 hidden_size(num_heads * head_size)를 병렬로 처리
    dim3 block(std::min(hidden_size, (int)THREADS_PER_BLOCK_DEFAULT));
    hipLaunchKernelGGL(transpose_for_scores_kernel_impl, grid, block, 0, stream,
                       output, input, batch_size, seq_len, num_heads, head_size);
    HIP_CHECK(hipGetLastError());
}

void launch_transpose_back_kernel(
    hipStream_t stream, float* output, const float* input,
    int batch_size, int seq_len, int num_heads, int head_size)
{
    if (batch_size * seq_len * num_heads * head_size == 0) return;
    int hidden_size = num_heads * head_size;
    dim3 grid(batch_size, seq_len);
    dim3 block(std::min(hidden_size, (int)THREADS_PER_BLOCK_DEFAULT));
    hipLaunchKernelGGL(transpose_back_kernel_impl, grid, block, 0, stream,
                       output, input, batch_size, seq_len, num_heads, head_size);
    HIP_CHECK(hipGetLastError());
}

void launch_untranspose_kernel(
    float* output_bnas, const float* input_bsh,
    int batch_size, int seq_len, int num_heads, int head_size, hipStream_t stream)
{
    if (batch_size * seq_len * num_heads * head_size == 0) return;
    int hidden_size = num_heads * head_size;
    dim3 grid(batch_size, seq_len);
    dim3 block(std::min(hidden_size, (int)THREADS_PER_BLOCK_DEFAULT));
    hipLaunchKernelGGL(untranspose_kernel_impl, grid, block, 0, stream,
                       output_bnas, input_bsh, batch_size, seq_len, num_heads, head_size, stream);
    HIP_CHECK(hipGetLastError());
}


// --- Attention Kernels ---
void launch_scale_and_mask_kernel(
    hipStream_t stream, float* attention_scores, const float* attention_mask,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k, float scale) 
{
    if (batch_size * num_heads * seq_len_q * seq_len_k == 0) return;

    // attention_scores [B, N, S_q, S_k]의 각 (B, N, S_q)에 대해 블록 할당
    dim3 grid(batch_size, num_heads, seq_len_q);
    // 블록 내 스레드들은 S_k 차원을 병렬 처리
    dim3 block(std::min(seq_len_k, (int)THREADS_PER_BLOCK_DEFAULT)); 

    hipLaunchKernelGGL(scale_and_mask_kernel_impl, grid, block, 0, stream,
                       attention_scores, attention_mask, batch_size, num_heads, seq_len_q, seq_len_k, scale);
    HIP_CHECK(hipGetLastError());
}

void launch_softmax_kernel(
    hipStream_t stream, float* output, const float* input,
    int M_rows, int N_softmax_dim)
{
    if (M_rows == 0 || N_softmax_dim == 0) return;

    // Softmax를 수행할 각 행(M_rows)에 대해 블록 할당
    dim3 grid(M_rows);
    // 블록 내 스레드들은 N_softmax_dim 차원을 병렬 처리
    dim3 block(std::min(N_softmax_dim, (int)THREADS_PER_BLOCK_DEFAULT));
    size_t shared_mem_size = block.x * sizeof(float);

    hipLaunchKernelGGL(softmax_kernel_impl, grid, block, shared_mem_size, stream,
                       output, input, M_rows, N_softmax_dim);
    HIP_CHECK(hipGetLastError());
}

void launch_softmax_backward_kernel(
    hipStream_t stream, float* grad_input, const float* grad_output, const float* output,
    int M_rows, int N_softmax_dim)
{
    if (M_rows == 0 || N_softmax_dim == 0) return;

    dim3 grid(M_rows); 
    dim3 block(std::min(N_softmax_dim, (int)THREADS_PER_BLOCK_DEFAULT));
    size_t shared_mem_size = block.x * sizeof(float);

    hipLaunchKernelGGL(softmax_backward_kernel_impl, grid, block, shared_mem_size, stream,
                       grad_input, grad_output, output,
                       M_rows, N_softmax_dim);
    HIP_CHECK(hipGetLastError());
}

// --- Element-wise Kernels ---
void launch_elementwise_add_kernel(hipStream_t stream, float* out, const float* in1, const float* in2, size_t num_elements) {
    if (num_elements == 0) return;
    dim3 block_dim(THREADS_PER_BLOCK_DEFAULT);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x);
    hipLaunchKernelGGL(elementwise_add_kernel_impl, grid_dim, block_dim, 0, stream, out, in1, in2, num_elements);
    HIP_CHECK(hipGetLastError());
}

void launch_accumulate_kernel(hipStream_t stream, float* target_and_out, const float* to_add, size_t num_elements) {
    if (num_elements == 0) return;
    dim3 block_dim(THREADS_PER_BLOCK_DEFAULT);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x);
    hipLaunchKernelGGL(accumulate_kernel_impl, grid_dim, block_dim, 0, stream, target_and_out, to_add, num_elements);
    HIP_CHECK(hipGetLastError());
}