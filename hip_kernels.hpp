#ifndef HIP_KERNELS_HPP
#define HIP_KERNELS_HPP

// 개선 사항 9: include 순서 정리 (Standard -> Third-party -> Project)
#include <cstddef>
#include <cfloat> // For FLT_MAX
#include <cmath>  // For M_PI_F
#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#include "common_hip.hpp"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif
#define HIP_LAUNCH_KERNEL(KERNEL_NAME, GRID, BLOCK, SHM, STREAM, ...)           \
    do {                                                                        \
        std::cout << "[KERNEL LAUNCH] " << #KERNEL_NAME << std::endl;           \
        hipLaunchKernelGGL(KERNEL_NAME, GRID, BLOCK, SHM, STREAM, __VA_ARGS__); \
        HIP_CHECK(hipGetLastError());                                           \
    } while (0)
// ============================================================================
// Constants
// ============================================================================
// 개선 사항 8: 하드코딩된 상수/매직넘버 정의
namespace KernelConstants {
    constexpr int   THREADS_PER_BLOCK_DEFAULT = 256;
    constexpr int   TRANSPOSE_BLOCK_DIM = 16; // Transpose 커널용 블록 크기
    constexpr int   TRANSPOSE_BLOCK_DIM_X = 16;
    constexpr int   TRANSPOSE_BLOCK_DIM_Y = 16;
    constexpr float ATTENTION_MASK_VALUE = -1.0e4f; // 어텐션 마스크에 사용할 큰 음수
}

// ============================================================================
// Device-side Utility Functions
// ============================================================================
// 개선 사항 1: 중복 정의 제거 및 공통 헤더로 이동

// ============================================================================
// Kernel Launchers Declarations
// ============================================================================

// Dropout (개선 사항 5: Philox RNG 사용)
void launch_dropout_forward(
    hipStream_t stream, float* output, float* mask, const float* input,
    size_t num_elements, float prob, float scale, unsigned long long seed, unsigned long long offset);

void launch_dropout_backward(
    hipStream_t stream, float* grad_input, const float* grad_output, const float* mask,
    size_t num_elements, float scale);

// Layer Normalization
void launch_layer_norm_forward_optimized(
    hipStream_t stream, float* out, float* mean, float* rstd,
    const float* inp, const float* gamma, const float* beta,
    int B, int C, float epsilon);

void launch_layer_norm_backward_optimized(
    hipStream_t stream,
    // [SUGGESTION] 파라미터 이름 명확화
    float* grad_input,
    float* grad_gamma, // d(gamma)
    float* grad_beta,  // d(beta)
    const float* grad_output,
    const float* input,
    const float* gamma,
    const float* mean,
    const float* rstd,
    int B,
    int C,
    float eps // [FIX] 안정성을 위한 Epsilon 값 추가
);

// GELU and Add_Bias_GELU
void launch_add_bias_gelu_kernel(
    hipStream_t stream, float* output, const float* input, const float* bias, int M, int N);

// (개선 사항 6: AtomicAdd contention 회피)
void launch_gelu_add_bias_backward_kernel(
    hipStream_t stream,
    float* grad_input_before_bias,
    float* grad_bias_part, // 블록별 부분 합을 저장할 임시 공간
    const float* grad_output_after_gelu,
    const float* input_before_gelu,
    int M, int N);

void launch_gelu_backward_kernel(
    hipStream_t stream, float* grad_input, const float* grad_output, const float* input, size_t num_elements);

void launch_gelu_forward_kernel(
    hipStream_t stream, float* output, const float* input, size_t num_elements);

// Reduction Kernels
void launch_reduce_sum_kernel(hipStream_t stream, float* out_vec, const float* in_matrix, int rows, int cols);
void reduce_sum_along_axis_0(hipStream_t stream, const GpuTensor& input, GpuTensor& output);
// Softmax Cross Entropy Loss
void launch_softmax_cross_entropy_loss_backward_optimized(
    hipStream_t stream, float* grad_logits, const float* logits,
    const int* labels, float* total_loss,
    int B, int S, int V, int ignore_index);

// AdamW Optimizer
void launch_adamw_update_kernel(
    hipStream_t stream, float* params, const float* grads, float* m, float* v,
    float lr, float beta1, float beta2, float eps, float weight_decay, int t, size_t num_elements);

// Utility Kernels
void launch_add_bias_kernel(
    hipStream_t stream, float* output, const float* input, const float* bias, int M, int N);

// Embedding Kernels
void launch_add_embeddings_kernel(
    hipStream_t stream, float* output, const int* input_ids, const int* token_type_ids,
    const float* word_embeddings, const float* position_embeddings,
    const float* token_type_embeddings, int batch_size, int seq_len,
    int hidden_size, int vocab_size, int max_position_embeddings);

// (개선 사항 6: AtomicAdd contention 회피)
void launch_embedding_backward_kernel(
    hipStream_t stream, float* grad_word_embeddings,
    const float* grad_output, const int* input_ids,
    int B, int S, int H, int V);

void launch_accumulate_positional_embedding_grad(
    hipStream_t stream, float* grad_pos_embeddings, const float* grad_output, int B, int S, int H);

void launch_accumulate_token_type_embedding_grad(
    hipStream_t stream, float* grad_token_type_embeddings,
    const float* grad_output, const int* token_type_ids,
    int B, int S, int H);

// Transpose Kernels for Attention
void launch_transpose_for_scores_kernel(
    hipStream_t stream, float* output, const float* input,
    int batch_size, int seq_len, int num_heads, int head_size);

void launch_transpose_back_kernel(
    hipStream_t stream, float* output, const float* input,
    int batch_size, int seq_len, int num_heads, int head_size);

// Attention Kernels (개선 사항 4: 마스크 일반화)
void launch_scale_and_mask_kernel(
    hipStream_t stream, float* attention_scores, const float* attention_mask,
    int B, int N, int Sq, int Sk, float scale);

void launch_softmax_kernel(
    hipStream_t stream, float* output, const float* input,
    int M_rows, int N_softmax_dim);

void launch_softmax_backward_kernel(
    hipStream_t stream, float* grad_input, const float* grad_output, const float* output,
    int M_rows, int N_softmax_dim);

// Element-wise Add/Accumulate Kernels
void launch_elementwise_add_kernel(hipStream_t stream, float* out, const float* in1, const float* in2, size_t num_elements);
void launch_accumulate_kernel(hipStream_t stream, float* target_and_out, const float* to_add, size_t num_elements);

// Element-wise Scale Kernel
void launch_scale_kernel(hipStream_t stream, float* data, float scale_factor, size_t num_elements);
void launch_set_identity_kernel(hipStream_t stream, float* matrix, int n);
void launch_elementwise_scale_kernel(hipStream_t stream, float* data, float scale, size_t num_elements);
void launch_elementwise_accumulate_kernel(hipStream_t stream, float* dst, const float* src, size_t num_elements);
void launch_add_diagonal_value_kernel(hipStream_t stream, float* matrix, int n, float value);
void launch_power_kernel(hipStream_t stream, float* data, float power, size_t num_elements);
void launch_matrix_scale_columns_kernel(hipStream_t stream, float* output, const float* input, const float* scales, int rows, int cols);
void launch_transpose_kernel(hipStream_t stream, float* dst, const float* src, int rows, int cols);
void launch_adamw_update_kernel(hipStream_t stream, float* weights, const float* gradients, float* m, float* v, 
                               float lr, float beta1, float beta2, float epsilon, float weight_decay, int t, size_t n);
                               void launch_add_bias_only_kernel(
    hipStream_t     stream,
    float*          output,     // in-place 가능
    const float*    input,
    const float*    bias,       // [N]
    int             M,          // rows
    int             N);         // cols
#endif // HIP_KERNELS_HPP