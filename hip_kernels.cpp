// =============================================================================
// hip_kernels.cpp — High-Performance, Safe HIP Kernel Implementations
// =============================================================================

#include "hip_kernels.hpp" // 프로젝트 헤더
#include <iomanip>  // std::setprecision
#include <algorithm> // std::min
// ============================================================================
// Device-side Utility & Reduction Functions
// ============================================================================

__device__ inline float gelu_fn_device(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI_F) * (x + 0.044715f * x * x * x)));
}
__device__ inline float gelu_grad_fn_device(float x) {
    const float cdf_constant = 0.044715f;
    const float sqrt_2_over_pi = sqrtf(2.0f / M_PI_F);
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + cdf_constant * x_cubed);
    float tanh_inner = tanhf(inner);
    float sech_inner_sq = 1.0f - tanh_inner * tanh_inner;
    float inner_derivative = sqrt_2_over_pi * (1.0f + 3.0f * cdf_constant * x * x);
    return 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech_inner_sq * inner_derivative;
}

__device__ inline float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__device__ inline float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

__device__ inline float blockReduceSum(float val) {
    extern __shared__ float smem[];
    const int lane    = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / warpSize)) ? smem[lane] : 0.0f;
    if (warp_id == 0) val = warpReduceSum(val);
    return val;
}

__device__ inline float blockReduceMax(float val) {
    extern __shared__ float smem[];
    const int lane    = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x / warpSize;

    val = warpReduceMax(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / warpSize)) ? smem[lane] : -FLT_MAX;
    if (warp_id == 0) val = warpReduceMax(val);
    return val;
}


// ============================================================================
// Kernel Implementations
// ============================================================================

__global__ void dropout_forward_kernel_impl(float* __restrict__ output, float* __restrict__ mask,
                                     const float* __restrict__ input, size_t num_elements,
                                     float prob, float scale,
                                     unsigned long long seed, unsigned long long offset) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    hiprandStatePhilox4_32_10_t state;
    hiprand_init(seed, idx, offset, &state);

    float rand_val = hiprand_uniform(&state);
    if (rand_val < prob) {
        mask[idx] = 0.0f;
        output[idx] = 0.0f;
    } else {
        mask[idx] = 1.0f;
        output[idx] = input[idx] * scale;
    }
}

__global__ void dropout_backward_kernel_impl(float* __restrict__ grad_input, const float* __restrict__ grad_output,
                                        const float* __restrict__ mask, size_t num_elements, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        grad_input[idx] = grad_output[idx] * mask[idx] * scale;
    }
}

__global__ void layer_norm_forward_kernel_optimized_impl(
    float* __restrict__ out, float* __restrict__ mean_out, float* __restrict__ rstd_out,
    const float* __restrict__ inp, const float* __restrict__ gamma, const float* __restrict__ beta,
    int B, int C, float epsilon)
{
    extern __shared__ float shared_data[];
    float* s_warp_results = shared_data;
    float* s_broadcast_params = &s_warp_results[(blockDim.x + warpSize - 1) / warpSize];

    int b = blockIdx.x;
    const float* inp_b = inp + b * C;
    float* out_b = out + b * C;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) thread_sum += inp_b[i];
    float total_sum = blockReduceSum(thread_sum);

    float mean;
    if (threadIdx.x == 0) {
        mean = total_sum / C;
        if(mean_out) mean_out[b] = mean;
        s_broadcast_params[0] = mean;
    }
    __syncthreads();
    mean = s_broadcast_params[0];

    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float val = inp_b[i] - mean;
        thread_sum_sq += val * val;
    }
    float total_sum_sq = blockReduceSum(thread_sum_sq);

    float rstd;
    if (threadIdx.x == 0) {
        float var = total_sum_sq / C;
        rstd = rsqrtf(var + epsilon);
        if(rstd_out) rstd_out[b] = rstd;
        s_broadcast_params[1] = rstd;
    }
    __syncthreads();
    rstd = s_broadcast_params[1];

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float n = (inp_b[i] - mean) * rstd;
        out_b[i] = n * gamma[i] + beta[i];
    }
}

__global__ void layer_norm_backward_kernel_optimized_impl(
    float* __restrict__ grad_input, float* __restrict__ grad_gamma_part, float* __restrict__ grad_beta_part,
    const float* __restrict__ grad_output, const float* __restrict__ input,
    const float* __restrict__ gamma, const float* __restrict__ mean, const float* __restrict__ rstd,
    int B, int C)
{
    extern __shared__ float shared_data[];
    float* s_warp_results = shared_data;
    float* s_broadcast_params = &s_warp_results[(blockDim.x + warpSize - 1) / warpSize];

    int b = blockIdx.x;
    const float* input_b = input + b * C;
    const float* grad_output_b = grad_output + b * C;
    float* grad_input_b = grad_input + b * C;
    const float mean_b = mean[b];
    const float rstd_b = rstd[b];

    float* grad_gamma_b = grad_gamma_part + b * C;
    float* grad_beta_b = grad_beta_part + b * C;

    float sum1_thread = 0.0f, sum2_thread = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const float x_hat_i = (input_b[i] - mean_b) * rstd_b;
        const float dL_dy_i = grad_output_b[i];
        grad_gamma_b[i] = dL_dy_i * x_hat_i;
        grad_beta_b[i] = dL_dy_i;
        
        const float dL_dy_gamma_i = dL_dy_i * gamma[i];
        sum1_thread += dL_dy_gamma_i;
        sum2_thread += dL_dy_gamma_i * x_hat_i;
    }

    const float sum1 = blockReduceSum(sum1_thread);
    const float sum2 = blockReduceSum(sum2_thread);

    if (threadIdx.x == 0) {
        s_broadcast_params[0] = sum1 / C;
        s_broadcast_params[1] = sum2 / C;
    }
    __syncthreads();

    const float c1 = s_broadcast_params[0];
    const float c2 = s_broadcast_params[1];

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const float x_hat_i = (input_b[i] - mean_b) * rstd_b;
        const float dL_dy_gamma_i = grad_output_b[i] * gamma[i];
        float dL_dx_i = dL_dy_gamma_i - c1 - (x_hat_i * c2);
        grad_input_b[i] = rstd_b * dL_dx_i;
    }
}

__global__ void add_bias_gelu_kernel_impl(float* output, const float* input, const float* bias, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int idx = row * N + col;
        output[idx] = gelu_fn_device(input[idx] + bias[col]);
    }
}

__global__ void add_bias_only_kernel_impl(float* out, const float* in, const float* bias, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        out[row * N + col] = in[row * N + col] + bias[col];
    }
}


__global__ void gelu_add_bias_backward_kernel_impl(
    float* grad_input_before_bias, float* grad_bias_part,
    const float* grad_output_after_gelu, const float* input_before_gelu,
    int M, int N)
{
    extern __shared__ float sdata[];
    int col = blockIdx.x;
    int tid_y = threadIdx.y;
    
    float my_sum = 0.0f;
    if (col < N) {
        for (int row = tid_y; row < M; row += blockDim.y) {
            int idx = row * N + col;
            float dL_dInputBeforeGelu = grad_output_after_gelu[idx] * gelu_grad_fn_device(input_before_gelu[idx]);
            grad_input_before_bias[idx] = dL_dInputBeforeGelu;
            my_sum += dL_dInputBeforeGelu;
        }
    }
    sdata[tid_y] = my_sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.y / 2; s > 0; s >>= 1) {
        if (tid_y < s) sdata[tid_y] += sdata[tid_y + s];
        __syncthreads();
    }

    if (tid_y == 0 && col < N) {
        grad_bias_part[col] = sdata[0];
    }
}

__global__ void gelu_backward_kernel_impl(float* grad_input, const float* grad_output, const float* input, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        grad_input[idx] = grad_output[idx] * gelu_grad_fn_device(input[idx]);
    }
}

__global__ void gelu_forward_kernel_impl(float* output, const float* input, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = gelu_fn_device(input[idx]);
    }
}

__global__ void reduce_sum_optimized_impl(float* out_vec, const float* in_matrix, int rows, int cols) {
    int col_idx = blockIdx.x;
    if (col_idx >= cols) return;
    
    const float* col_ptr = in_matrix + col_idx;
    float thread_sum = 0.0f;

    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        thread_sum += col_ptr[i * cols];
    }
    
    float total_col_sum = blockReduceSum(thread_sum);

    if (threadIdx.x == 0) {
        out_vec[col_idx] = total_col_sum;
    }
}

__global__ void softmax_cross_entropy_loss_backward_optimized_impl(
    float* grad_logits, const float* logits, const int* labels,
    float* total_loss, int B, int S, int V, int ignore_index) {
    extern __shared__ float shared_data[];
    float* s_warp_results = shared_data;
    float* s_broadcast_params = &s_warp_results[(blockDim.x + warpSize - 1) / warpSize];

    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int seq_idx = b * S + s;
    const int label_val = labels[seq_idx];

    if (label_val == ignore_index) {
        for (int i = tid; i < V; i += blockDim.x) {
            grad_logits[seq_idx * V + i] = 0.0f;
        }
        return;
    }

    float thread_max = -FLT_MAX;
    for (int i = tid; i < V; i += blockDim.x) thread_max = fmaxf(thread_max, logits[seq_idx * V + i]);
    float max_logit = blockReduceMax(thread_max);

    if (tid == 0) s_broadcast_params[0] = max_logit;
    __syncthreads();
    max_logit = s_broadcast_params[0];

    float thread_sum_exp = 0.0f;
    for (int i = tid; i < V; i += blockDim.x) thread_sum_exp += expf(logits[seq_idx * V + i] - max_logit);
    float sum_exp = blockReduceSum(thread_sum_exp);

    if (tid == 0) {
        s_broadcast_params[0] = sum_exp;
        const float logit_label = logits[seq_idx * V + label_val];
        const float log_prob = (logit_label - max_logit) - logf(sum_exp);
        atomicAdd(total_loss, -log_prob);
    }
    __syncthreads();
    sum_exp = s_broadcast_params[0];

    const float inv_sum_exp = 1.0f / fmaxf(sum_exp, 1e-9f);
    for (int i = tid; i < V; i += blockDim.x) {
        const float prob = expf(logits[seq_idx * V + i] - max_logit) * inv_sum_exp;
        const float grad = (i == label_val) ? (prob - 1.0f) : prob;
        grad_logits[seq_idx * V + i] = grad;
    }
}

__global__ void adamw_update_kernel_impl(float* params, const float* grads, float* m, float* v,
                                  float lr, float beta1, float beta2, float eps,
                                  float weight_decay, int t, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float p = params[idx];
        const float g = grads[idx];

        p -= lr * weight_decay * p;

        float m_t = beta1 * m[idx] + (1.0f - beta1) * g;
        float v_t = beta2 * v[idx] + (1.0f - beta2) * g * g;
        m[idx] = m_t;
        v[idx] = v_t;

        float m_hat = m_t / (1.0f - powf(beta1, t));
        float v_hat = v_t / (1.0f - powf(beta2, t));

        p -= lr * m_hat / (sqrtf(v_hat) + eps);
        params[idx] = p;
    }
}

__global__ void add_embeddings_kernel_impl(float* output, const int* input_ids, const int* token_type_ids,
                                     const float* word_embeddings, const float* position_embeddings,
                                     const float* token_type_embeddings, int B, int S,
                                     int H, int vocab_size, int max_pos_embed) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int h_tid = threadIdx.x;

    if (b < B && s < S && h_tid < H) {
        int seq_item_idx = b * S + s;
        int output_base_idx = seq_item_idx * H;

        int word_id = std::max(0, std::min(input_ids[seq_item_idx], vocab_size - 1));
        int token_type_id = std::max(0, std::min(token_type_ids[seq_item_idx], 1));
        int pos_id = std::max(0, std::min(s, max_pos_embed - 1));

        float word_emb_val = word_embeddings[word_id * H + h_tid];
        float pos_emb_val = position_embeddings[pos_id * H + h_tid];
        float token_type_emb_val = token_type_embeddings[token_type_id * H + h_tid];

        output[output_base_idx + h_tid] = word_emb_val + pos_emb_val + token_type_emb_val;
    }
}

__global__ void embedding_backward_atomic_impl(
    float* grad_word_embeddings, const float* grad_output,
    const int* input_ids, size_t num_tokens, int H) 
{
    size_t token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    int word_id = input_ids[token_idx];
    if (word_id < 0) return;

    const float* grad_out_vec = grad_output + token_idx * H;
    float* grad_word_emb_vec = grad_word_embeddings + (size_t)word_id * H;

    for (int h = 0; h < H; ++h) {
        atomicAdd(&grad_word_emb_vec[h], grad_out_vec[h]);
    }
}

__global__ void accumulate_positional_embedding_grad_impl(
    float* grad_pos_embeddings, const float* grad_output, int B, int S, int H)
{
    int pos_id = blockIdx.x;
    int h_tid = threadIdx.x;

    if (pos_id < S && h_tid < H) {
        float sum = 0.0f;
        for (int b = 0; b < B; ++b) {
            sum += grad_output[(b * S + pos_id) * H + h_tid];
        }
        grad_pos_embeddings[pos_id * H + h_tid] = sum;
    }
}

__global__ void accumulate_token_type_embedding_grad_impl(
    float* grad_token_type_embeddings, const float* grad_output,
    const int* token_type_ids, int B, int S, int H)
{
    int type_id = blockIdx.x;
    int h_tid = threadIdx.x;

    if (type_id < 2 && h_tid < H) {
        float sum = 0.0f;
        for (int i = 0; i < B * S; ++i) {
            if (token_type_ids[i] == type_id) {
                sum += grad_output[i * H + h_tid];
            }
        }
        grad_token_type_embeddings[type_id * H + h_tid] = sum;
    }
}


__global__ void transpose_for_scores_kernel_impl(float* output, const float* input,
                                           int B, int S, int N, int A) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int h_flat_idx = threadIdx.x;

    if (b < B && s < S && h_flat_idx < (N * A)) {
        int n_out = h_flat_idx / A;
        int a_out = h_flat_idx % A;
        int input_idx = (b * S + s) * (N * A) + h_flat_idx;
        int output_idx = ((b * N + n_out) * S + s) * A + a_out;
        output[output_idx] = input[input_idx];
    }
}

__global__ void transpose_back_kernel_impl(float* output, const float* input,
                                     int B, int S, int N, int A) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int h_flat_idx = threadIdx.x;

    if (b < B && s < S && h_flat_idx < (N * A)) {
        int n_in = h_flat_idx / A;
        int a_in = h_flat_idx % A;
        int input_idx = ((b * N + n_in) * S + s) * A + a_in;
        int output_idx = (b * S + s) * (N * A) + h_flat_idx;
        output[output_idx] = input[output_idx];
    }
}

__global__ void scale_and_mask_kernel_impl(float* scores, const float* mask,
                                     int B, int N, int Sq, int Sk, float scale) {
    int b = blockIdx.x;
    int n = blockIdx.y;
    int sq = blockIdx.z;
    int sk = threadIdx.x;

    if (b < B && n < N && sq < Sq && sk < Sk) {
        int score_idx = ((b * N + n) * Sq + sq) * Sk + sk;
        float add_val = (mask != nullptr) ? mask[b * Sk + sk] : 0.0f;
        scores[score_idx] = scores[score_idx] * scale + add_val;
    }
}

__global__ void softmax_kernel_impl(float* output, const float* input, int M, int N_softmax_dim) {
    extern __shared__ float sdata[];
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (row_idx >= M) return;

    const float* row_input = input + row_idx * N_softmax_dim;
    float* row_output = output + row_idx * N_softmax_dim;

    float max_val = -FLT_MAX;
    for (int i = tid; i < N_softmax_dim; i += blockDim.x) max_val = fmaxf(max_val, row_input[i]);
    max_val = blockReduceMax(max_val);

    float sum_exp = 0.0f;
    for (int i = tid; i < N_softmax_dim; i += blockDim.x) sum_exp += expf(row_input[i] - max_val);
    sum_exp = blockReduceSum(sum_exp);

    sum_exp = fmaxf(sum_exp, 1e-9f);
    for (int i = tid; i < N_softmax_dim; i += blockDim.x) {
        row_output[i] = expf(row_input[i] - max_val) / sum_exp;
    }
}

__global__ void softmax_backward_kernel_impl(
    float* grad_input, const float* grad_output, const float* output,
    int M, int N_softmax_dim)
{
    extern __shared__ float sdata[];
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (row_idx >= M) return;

    const float* row_grad_output = grad_output + row_idx * N_softmax_dim;
    const float* row_output = output + row_idx * N_softmax_dim;
    float* row_grad_input = grad_input + row_idx * N_softmax_dim;

    float sum_val = 0.0f;
    for (int j = tid; j < N_softmax_dim; j += blockDim.x) sum_val += row_grad_output[j] * row_output[j];
    sum_val = blockReduceSum(sum_val);

    for (int i = tid; i < N_softmax_dim; i += blockDim.x) {
        row_grad_input[i] = row_output[i] * (row_grad_output[i] - sum_val);
    }
}

__global__ void elementwise_add_kernel_impl(float* out, const float* in1, const float* in2, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in1[idx] + in2[idx];
}

__global__ void accumulate_kernel_impl(float* target_and_out, const float* to_add, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // idx가 n보다 작은 경우에만 누적 연산 수행
    if (idx < n) {
        target_and_out[idx] += to_add[idx];  // 누적 연산
    }
}


__global__ void scale_kernel_impl(float* data, float scale_factor, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) data[idx] *= scale_factor;
}

__global__ void set_identity_kernel_impl(float* matrix, int n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (size_t)n) matrix[idx * n + idx] = 1.0f;
}

__global__ void elementwise_scale_kernel_impl(float* data, float scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= scale;
}

__global__ void elementwise_accumulate_kernel_impl(float* dst, const float* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += src[idx];
}

__global__ void add_diagonal_value_kernel_impl(float* matrix, int n, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (size_t)n) matrix[idx * n + idx] += value;
}

__global__ void power_kernel_impl(float* data, float power, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = powf(data[idx], power);
}

__global__ void matrix_scale_columns_corrected_impl(float* output, const float* input, const float* scales, int rows, int cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)rows * cols) return;
    int c = idx % cols;
    output[idx] = input[idx] * scales[c];
}

__global__ void transpose_kernel(float* dst, const float* src, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        dst[c * rows + r] = src[r * cols + c];
    }
}


// ============================================================================
// Kernel Launcher Implementations (Optimized)
// ============================================================================
void launch_dropout_forward(
    hipStream_t stream, float* output, float* mask, const float* input,
    size_t num_elements, float prob, float scale, unsigned long long seed, unsigned long long offset)
{
    if (num_elements == 0) return;
    dim3 block_dim(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x);
    HIP_LAUNCH_KERNEL(dropout_forward_kernel_impl, grid_dim, block_dim, 0, stream,
                       output, mask, input, num_elements, prob, scale, seed, offset);
}

void launch_dropout_backward(
    hipStream_t stream, float* grad_input, const float* grad_output, const float* mask,
    size_t num_elements, float scale)
{
    if (num_elements == 0) return;
    dim3 block_dim(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x);
    HIP_LAUNCH_KERNEL(dropout_backward_kernel_impl, grid_dim, block_dim, 0, stream,
                       grad_input, grad_output, mask, num_elements, scale);
}

void launch_layer_norm_forward_optimized(
    hipStream_t stream, float* out, float* mean, float* rstd,
    const float* inp, const float* gamma, const float* beta,
    int B, int C, float epsilon)
{
    if (B == 0 || C == 0) return;
    dim3 grid(B);
    dim3 block(std::min(C, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
    size_t shared_mem_size = ((block.x + warpSize - 1) / warpSize) * sizeof(float) + 2 * sizeof(float);
    HIP_LAUNCH_KERNEL(layer_norm_forward_kernel_optimized_impl, grid, block, shared_mem_size, stream,
                       out, mean, rstd, inp, gamma, beta, B, C, epsilon);
}

void launch_layer_norm_backward_optimized(
    hipStream_t stream, float* grad_input, float* grad_gamma_part, float* grad_beta_part,
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* rstd,
    int B, int C)
{
    printf("Launching layer_norm_backward_optimized with B=%d, C=%d\n", B, C);

    if (B == 0 || C == 0) return;

    // 배치 차원과 채널 차원 분리
    int block_size = std::min(C, KernelConstants::THREADS_PER_BLOCK_DEFAULT); // 채널에 맞는 스레드 수
    dim3 block(block_size);
    dim3 grid((B + block.x - 1) / block.x);  // 배치 차원을 나누어서 grid 크기 설정

    size_t shared_mem_size = (block.x + warpSize - 1) / warpSize * sizeof(float) * 2;
    
    // 커널 호출
    HIP_LAUNCH_KERNEL(layer_norm_backward_kernel_optimized_impl, grid, block, shared_mem_size, stream,
                       grad_input, grad_gamma_part, grad_beta_part,
                       grad_output, input, gamma, mean, rstd, B, C);

    printf("Finished launching layer_norm_backward_optimized\n");
}


void launch_add_bias_gelu_kernel(hipStream_t stream, float* output, const float* input, const float* bias, int M, int N) {
    if(M==0||N==0)return;
    dim3 threads(KernelConstants::TRANSPOSE_BLOCK_DIM_X, KernelConstants::TRANSPOSE_BLOCK_DIM_Y);
    dim3 grid((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    HIP_LAUNCH_KERNEL(add_bias_gelu_kernel_impl, grid, threads, 0, stream, output, input, bias, M, N);
}

void launch_gelu_add_bias_backward_kernel(
    hipStream_t stream,
    float* grad_input_before_bias,
    float* grad_bias_part,
    const float* grad_output_after_gelu,
    const float* input_before_gelu,
    int M, int N)
{
    if (M == 0 || N == 0) return;
    dim3 threads(1, 256);
    dim3 grid(N, 1);
    size_t shared_mem_size = threads.y * sizeof(float);
    HIP_LAUNCH_KERNEL(gelu_add_bias_backward_kernel_impl, grid, threads, shared_mem_size, stream,
                       grad_input_before_bias, grad_bias_part,
                       grad_output_after_gelu, input_before_gelu, M, N);
}

void launch_gelu_backward_kernel(hipStream_t stream, float* grad_input, const float* grad_output, const float* input, size_t num_elements) {
    if(num_elements==0)return;
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((num_elements+block.x-1)/block.x);
    HIP_LAUNCH_KERNEL(gelu_backward_kernel_impl, grid, block, 0, stream, grad_input, grad_output, input, num_elements);
}

void launch_gelu_forward_kernel(hipStream_t stream, float* output, const float* input, size_t num_elements) {
    if(num_elements==0)return;
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((num_elements+block.x-1)/block.x);
    HIP_LAUNCH_KERNEL(gelu_forward_kernel_impl, grid, block, 0, stream, output, input, num_elements);
}

void launch_reduce_sum_kernel(hipStream_t stream, float* out_vec, const float* in_matrix, int rows, int cols) {
    if (rows == 0 || cols == 0) return;

    // 그리드와 블록 크기 설정
    dim3 grid(cols);  // 열 단위로 그리드 크기 설정
    dim3 block(std::min(rows, KernelConstants::THREADS_PER_BLOCK_DEFAULT));  // 행 단위로 블록 크기 설정

    // 공유 메모리 크기 설정 (최적화)
    size_t shared_mem_size = ((block.x + warpSize - 1) / warpSize) * sizeof(float);

    // 커널 실행
    HIP_LAUNCH_KERNEL(reduce_sum_optimized_impl, grid, block, shared_mem_size, stream, out_vec, in_matrix, rows, cols);
}

void launch_softmax_cross_entropy_loss_backward_optimized(
    hipStream_t stream, float* grad_logits, const float* logits,
    const int* labels, float* total_loss,
    int B, int S, int V, int ignore_index)
{
    if (B == 0 || S == 0 || V == 0) return;
    dim3 grid(B, S);
    dim3 block(std::min(V, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
    size_t shared_mem_size = ((block.x + warpSize - 1) / warpSize) * sizeof(float) + 2 * sizeof(float);
    HIP_LAUNCH_KERNEL(softmax_cross_entropy_loss_backward_optimized_impl, grid, block, shared_mem_size, stream,
                       grad_logits, logits, labels, total_loss, B, S, V, ignore_index);
}

void launch_adamw_update_kernel(hipStream_t stream, float* params, const float* grads, float* m, float* v,
                               float lr, float beta1, float beta2, float eps, float weight_decay, int t, size_t n) {
    if (n==0) return;
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((n + block.x - 1) / block.x);
    HIP_LAUNCH_KERNEL(adamw_update_kernel_impl, grid, block, 0, stream,
                       params, grads, m, v, lr, beta1, beta2, eps, weight_decay, t, n);
}

void launch_add_bias_kernel(
    hipStream_t stream, float* output, const float* input, const float* bias, int M, int N)
{
    if (M == 0 || N == 0) return;
    dim3 threads(KernelConstants::TRANSPOSE_BLOCK_DIM_X, KernelConstants::TRANSPOSE_BLOCK_DIM_Y);
    dim3 grid((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    HIP_LAUNCH_KERNEL(add_bias_gelu_kernel_impl, grid, threads, 0, stream, output, input, bias, M, N);
}

void launch_add_bias_only_kernel(hipStream_t stream, float* output, const float* input, const float* bias, int M, int N)
{
    if (M == 0 || N == 0) return;
    dim3 block(KernelConstants::TRANSPOSE_BLOCK_DIM_X, KernelConstants::TRANSPOSE_BLOCK_DIM_Y);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    HIP_LAUNCH_KERNEL(add_bias_only_kernel_impl, grid, block, 0, stream, output, input, bias, M, N);
}


void launch_add_embeddings_kernel(
    hipStream_t stream, float* output, const int* input_ids, const int* token_type_ids,
    const float* word_embeddings, const float* position_embeddings,
    const float* token_type_embeddings, int batch_size, int seq_len,
    int hidden_size, int vocab_size, int max_position_embeddings)
{
    if (batch_size == 0 || seq_len == 0 || hidden_size == 0) return;
    dim3 grid(batch_size, seq_len);
    dim3 block(std::min(hidden_size, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
    HIP_LAUNCH_KERNEL(add_embeddings_kernel_impl, grid, block, 0, stream,
                       output, input_ids, token_type_ids, word_embeddings, position_embeddings,
                       token_type_embeddings, batch_size, seq_len, hidden_size, vocab_size, max_position_embeddings);
}

void launch_embedding_backward_kernel(
    hipStream_t stream, float* grad_word_embeddings,
    const float* grad_output, const int* input_ids,
    int B, int S, int H, int V)
{
    if (B * S == 0) return;
    size_t num_tokens = (size_t)B * S;
    dim3 block(256);
    dim3 grid((num_tokens + block.x - 1) / block.x);
    
    HIP_LAUNCH_KERNEL(embedding_backward_atomic_impl, grid, block, 0, stream,
                       grad_word_embeddings, grad_output, input_ids, num_tokens, H);
}

void launch_accumulate_positional_embedding_grad(
    hipStream_t stream, float* grad_pos_embeddings, const float* grad_output, int B, int S, int H)
{
    if (B*S*H == 0) return;
    dim3 grid(S);
    dim3 block(std::min(H, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
    HIP_LAUNCH_KERNEL(accumulate_positional_embedding_grad_impl, grid, block, 0, stream,
                       grad_pos_embeddings, grad_output, B, S, H);
}

void launch_accumulate_token_type_embedding_grad(
    hipStream_t stream, float* grad_token_type_embeddings,
    const float* grad_output, const int* token_type_ids,
    int B, int S, int H)
{
    if (B*S*H == 0) return;
    dim3 grid(2);
    dim3 block(std::min(H, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
    HIP_LAUNCH_KERNEL(accumulate_token_type_embedding_grad_impl, grid, block, 0, stream,
                       grad_token_type_embeddings, grad_output, token_type_ids, B, S, H);
}

void launch_transpose_for_scores_kernel(
    hipStream_t stream, float* output, const float* input,
    int batch_size, int seq_len, int num_heads, int head_size)
{
    if (batch_size * seq_len * num_heads * head_size == 0) return;
    int hidden_size = num_heads * head_size;
    dim3 grid(batch_size, seq_len);
    int block_dim = std::min(hidden_size, KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 block(block_dim);
    HIP_LAUNCH_KERNEL(transpose_for_scores_kernel_impl, grid, block, 0, stream,
                       output, input, batch_size, seq_len, num_heads, head_size);
}

void launch_transpose_back_kernel(
    hipStream_t stream, float* output, const float* input,
    int batch_size, int seq_len, int num_heads, int head_size)
{
    if (batch_size * seq_len * num_heads * head_size == 0) return;
    int hidden_size = num_heads * head_size;
    dim3 grid(batch_size, seq_len);
    int block_dim = std::min(hidden_size, KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 block(block_dim);
    HIP_LAUNCH_KERNEL(transpose_back_kernel_impl, grid, block, 0, stream,
                       output, input, batch_size, seq_len, num_heads, head_size);
}

void launch_scale_and_mask_kernel(
    hipStream_t stream, float* attention_scores, const float* attention_mask,
    int B, int N, int Sq, int Sk, float scale)
{
    if (B * N * Sq * Sk == 0) return;
    dim3 grid(B, N, Sq);
    dim3 block(std::min(Sk, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
    HIP_LAUNCH_KERNEL(scale_and_mask_kernel_impl, grid, block, 0, stream,
                       attention_scores, attention_mask, B, N, Sq, Sk, scale);
}

void launch_softmax_kernel(
    hipStream_t stream, float* output, const float* input,
    int M_rows, int N_softmax_dim)
{
    if (M_rows == 0 || N_softmax_dim == 0) return;
    dim3 grid(M_rows);
    dim3 block(std::min(N_softmax_dim, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
    size_t shared_mem_size = ((block.x + warpSize - 1) / warpSize) * sizeof(float);
    HIP_LAUNCH_KERNEL(softmax_kernel_impl, grid, block, shared_mem_size, stream,
                       output, input, M_rows, N_softmax_dim);
}

void launch_softmax_backward_kernel(
    hipStream_t stream, float* grad_input, const float* grad_output, const float* output,
    int M_rows, int N_softmax_dim)
{
    if (M_rows == 0 || N_softmax_dim == 0) return;
    dim3 grid(M_rows);
    dim3 block(std::min(N_softmax_dim, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
    size_t shared_mem_size = ((block.x + warpSize - 1) / warpSize) * sizeof(float);
    HIP_LAUNCH_KERNEL(softmax_backward_kernel_impl, grid, block, shared_mem_size, stream,
                       grad_input, grad_output, output, M_rows, N_softmax_dim);
}

void launch_elementwise_add_kernel(hipStream_t stream, float* out, const float* in1, const float* in2, size_t num_elements) {
    if (num_elements == 0) return;
    dim3 block_dim(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x);
    HIP_LAUNCH_KERNEL(elementwise_add_kernel_impl, grid_dim, block_dim, 0, stream, out, in1, in2, num_elements);
}

__host__ void validate_memory_access(float* ptr, size_t size, const char* name) {
    hipPointerAttribute_t attr;
    hipError_t err = hipPointerGetAttributes(&attr, ptr);
    
    if (err != hipSuccess) {
        printf("[ERROR] %s: 포인터 속성 조회 실패: %s\n", name, hipGetErrorString(err));
        return;
    }
    
    printf("[MEMORY] %s: ptr=%p, type=%d, device=%d\n", 
           name, ptr, attr.type, attr.device);
    
    // 메모리 범위 검사
    char test_byte;
    err = hipMemcpy(&test_byte, ptr, 1, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        printf("[ERROR] %s: 메모리 읽기 실패 (시작): %s\n", name, hipGetErrorString(err));
    }
    
    err = hipMemcpy(&test_byte, ptr + size - 1, 1, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        printf("[ERROR] %s: 메모리 읽기 실패 (끝): %s\n", name, hipGetErrorString(err));
    }
}
void launch_accumulate_kernel(hipStream_t stream, float* target_and_out, const float* to_add, size_t num_elements) {
    if (num_elements == 0) return;
    
    std::cout << "[DEBUG] accumulate_kernel 시작" << std::endl;
    std::cout << "[DEBUG] num_elements = " << num_elements << std::endl;
    std::cout << "[DEBUG] target_and_out pointer = " << target_and_out << std::endl;
    std::cout << "[DEBUG] to_add pointer = " << to_add << std::endl;
    
    // 입력 값 확인 (처음 10개)
    float* host_target = new float[std::min(num_elements, (size_t)10)];
    float* host_to_add = new float[std::min(num_elements, (size_t)10)];
    
    HIP_CHECK(hipMemcpy(host_target, target_and_out, std::min(num_elements, (size_t)10) * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_to_add, to_add, std::min(num_elements, (size_t)10) * sizeof(float), hipMemcpyDeviceToHost));
    
    std::cout << "[DEBUG] 커널 실행 전 값들:" << std::endl;
    for (int i = 0; i < std::min(num_elements, (size_t)10); ++i) {
        std::cout << "[DEBUG] target[" << i << "] = " << std::fixed << std::setprecision(6) << host_target[i] << std::endl;
    }
    for (int i = 0; i < std::min(num_elements, (size_t)10); ++i) {
        std::cout << "[DEBUG] to_add[" << i << "] = " << std::fixed << std::setprecision(6) << host_to_add[i] << std::endl;
    }
    
    // 블록과 그리드 크기 계산
    dim3 block_dim(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x);
    
    // 메모리 유효성 검사
    size_t bytes = num_elements * sizeof(float);
    validate_memory_access(target_and_out, bytes, "target_and_out");
    validate_memory_access((float*)to_add, bytes, "to_add");
    
    std::cout << "[DEBUG] grid_dim = " << grid_dim.x << ", block_dim = " << block_dim.x << std::endl;
    
    // 커널 실행
    std::cout << "[DEBUG] accumulate_kernel_impl 실행 중..." << std::endl;
    HIP_LAUNCH_KERNEL(accumulate_kernel_impl, grid_dim, block_dim, 0, stream, 
                       target_and_out, to_add, num_elements);
    
    // 동기화
    HIP_CHECK(hipStreamSynchronize(stream));
    std::cout << "[DEBUG] 커널 실행 완료" << std::endl;
    
    // 결과 확인 (처음 10개)
    float* host_result = new float[std::min(num_elements, (size_t)10)];
    HIP_CHECK(hipMemcpy(host_result, target_and_out, std::min(num_elements, (size_t)10) * sizeof(float), hipMemcpyDeviceToHost));
    
    std::cout << "[DEBUG] 커널 실행 후 결과:" << std::endl;
    for (int i = 0; i < std::min(num_elements, (size_t)10); ++i) {
        std::cout << "[DEBUG] result[" << i << "] = " << std::fixed << std::setprecision(6) << host_result[i] 
                  << " (expected: " << (host_target[i] + host_to_add[i]) << ")" << std::endl;
    }
    
    // 메모리 해제
    delete[] host_target;
    delete[] host_to_add;
    delete[] host_result;
    
    std::cout << "[DEBUG] accumulate_kernel 완료" << std::endl;
}






void launch_scale_kernel(hipStream_t stream, float* data, float scale_factor, size_t n) {
    if(n==0)return;
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((n + block.x - 1) / block.x);
    HIP_LAUNCH_KERNEL(scale_kernel_impl, grid, block, 0, stream, data, scale_factor, n);
}

void launch_set_identity_kernel(hipStream_t stream, float* matrix, int n) {
    if (n==0) return;
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((n + block.x - 1) / block.x);
    HIP_LAUNCH_KERNEL(set_identity_kernel_impl, grid, block, 0, stream, matrix, n);
}

void launch_elementwise_scale_kernel(hipStream_t stream, float* data, float scale, size_t num_elements) {
    if (num_elements == 0) return;
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((num_elements + block.x - 1) / block.x);
    HIP_LAUNCH_KERNEL(elementwise_scale_kernel_impl, grid, block, 0, stream, data, scale, num_elements);
}

void launch_elementwise_accumulate_kernel(hipStream_t stream, float* dst, const float* src, size_t num_elements) {
    if (num_elements == 0) return;
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((num_elements + block.x - 1) / block.x);
    HIP_LAUNCH_KERNEL(elementwise_accumulate_kernel_impl, grid, block, 0, stream, dst, src, num_elements);
}

void launch_add_diagonal_value_kernel(hipStream_t stream, float* matrix, int n, float value) {
    if (n==0) return;
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((n + block.x - 1) / block.x);
    HIP_LAUNCH_KERNEL(add_diagonal_value_kernel_impl, grid, block, 0, stream, matrix, n, value);
}

void launch_power_kernel(hipStream_t stream, float* data, float power, size_t num_elements) {
    if(num_elements==0)return;
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((num_elements + block.x - 1) / block.x);
    HIP_LAUNCH_KERNEL(power_kernel_impl, grid, block, 0, stream, data, power, num_elements);
}

void launch_matrix_scale_columns_kernel(hipStream_t stream, float* output, const float* input, const float* scales, int rows, int cols) {
    if (rows == 0 || cols == 0) return;
    size_t num_elements = (size_t)rows * cols;
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((num_elements + block.x - 1) / block.x);
    HIP_LAUNCH_KERNEL(matrix_scale_columns_corrected_impl, grid, block, 0, stream, output, input, scales, rows, cols);
}

void launch_transpose_kernel(hipStream_t stream, float* dst, const float* src, int rows, int cols) {
    if (rows==0 || cols==0) return;
    dim3 block(KernelConstants::TRANSPOSE_BLOCK_DIM_X, KernelConstants::TRANSPOSE_BLOCK_DIM_Y);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    HIP_LAUNCH_KERNEL(transpose_kernel, grid, block, 0, stream, dst, src, rows, cols);
}
