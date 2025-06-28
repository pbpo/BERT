// 개선 사항 9: include 순서 정리 (Standard -> Third-party -> Project)
#include <algorithm> // For std::min, std::max
#include <cfloat>    // For FLT_MAX

#include "hip_kernels.hpp"

// ============================================================================
// Device-side Reduction Helpers
// ============================================================================
// 개선 사항 2: __shfl_down -> __shfl_down_sync
__device__ float warpReduceSum(float val) {
    const int all_lanes = 0xFFFFFFFF;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);  // _sync 제거
    }
    return val;
}

__device__ float warpReduceMax(float val) {
    const int all_lanes = 0xFFFFFFFF;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset));  // _sync 제거
    }
    return val;
}

__device__ inline float blockReduceSum(float val) {
    // 개선 사항 3: 공유 메모리 크기 산정
    extern __shared__ float s_warp_sums[];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) {
        s_warp_sums[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / warpSize)) ? s_warp_sums[lane] : 0.0f;
    if (warp_id == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

__device__ inline float blockReduceMax(float val) {
    extern __shared__ float s_warp_maxes[];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    val = warpReduceMax(val);

    if (lane == 0) {
        s_warp_maxes[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / warpSize)) ? s_warp_maxes[lane] : -FLT_MAX;
    if (warp_id == 0) {
        val = warpReduceMax(val);
    }
    return val;
}


// ============================================================================
// HIP Kernel Implementations
// ============================================================================
__global__ void dropout_forward_kernel_impl(float* output, const float* input, float* mask,
                                     size_t num_elements, float prob, float scale,
                                     unsigned long long seed, unsigned long long offset) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // 개선 사항 5: Philox RNG 사용으로 성능 향상
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

__global__ void dropout_backward_kernel_impl(float* grad_input, const float* grad_output,
                                        const float* mask, size_t num_elements, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        grad_input[idx] = grad_output[idx] * mask[idx] * scale;
    }
}


__global__ void layer_norm_forward_kernel_warp_optimized_impl(
    float* out, float* mean_out, float* rstd_out,
    const float* inp, const float* gamma, const float* beta,
    int B, int C, float epsilon)
{
    extern __shared__ float shared_data[];
    float* s_warp_results = shared_data;
    // 개선 사항 3: s_broadcast_params의 위치를 정확히 계산
    float* s_broadcast_params = &s_warp_results[(blockDim.x + warpSize - 1) / warpSize];

    int b = blockIdx.x;
    const float* inp_b = inp + b * C;
    float* out_b = out + b * C;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        thread_sum += inp_b[i];
    }
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

// 개선 사항 6: atomicAdd 경쟁을 피하기 위해 부분 합(partial sum) 사용
__global__ void layer_norm_backward_kernel_optimized_impl(
    float* grad_input, float* grad_gamma_part, float* grad_beta_part,
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* rstd,
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

    float sum1_thread = 0.0f;
    float sum2_thread = 0.0f;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const float x_hat_i = (input_b[i] - mean_b) * rstd_b;
        const float dL_dy_i = grad_output_b[i];
        grad_gamma_b[i] = dL_dy_i * x_hat_i;
        grad_beta_b[i] = dL_dy_i;
        
        const float dL_dy_gamma_i = dL_dy_i * gamma[i];
        sum1_thread += dL_dy_gamma_i;
        sum2_thread += dL_dy_gamma_i * x_hat_i;
    }
    __syncthreads();

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
        const float dL_dy_i = grad_output_b[i];
        const float dL_dy_gamma_i = dL_dy_i * gamma[i];
        float dL_dx_i = dL_dy_gamma_i - c1 - (x_hat_i * c2);
        grad_input_b[i] = rstd_b * dL_dx_i;
    }
}

__global__ void add_bias_gelu_kernel_impl(float* output, const float* input, const float* bias, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int idx = row * N + col;
        float val = input[idx] + bias[col];
        output[idx] = gelu_fn_device(val);
    }
}

__global__ void gelu_add_bias_backward_kernel_impl(
    float* grad_input_before_bias,
    float* grad_bias_part,
    const float* grad_output_after_gelu,
    const float* input_before_gelu,
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
        if (tid_y < s) {
            sdata[tid_y] += sdata[tid_y + s];
        }
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

__global__ void reduce_sum_kernel_impl(float* out_vec, const float* in_matrix, int rows, int cols) {
    extern __shared__ float sdata[];
    int j = blockIdx.x; // column index, which is the target index in out_vec
    int tid = threadIdx.x;

    float sum = 0.0f;
    if (j < cols) {
        for (int i = tid; i < rows; i += blockDim.x) {
            sum += in_matrix[i * cols + j];
        }
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && j < cols) {
        out_vec[j] = sdata[0];
    }
}

__global__ void scale_kernel_impl(float* data, float scale_factor, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] *= scale_factor;
    }
}

__global__ void softmax_cross_entropy_loss_backward_kernel_optimized_impl(
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
    for (int i = tid; i < V; i += blockDim.x) {
        thread_max = fmaxf(thread_max, logits[seq_idx * V + i]);
    }
    float max_logit = blockReduceMax(thread_max);

    if (tid == 0) s_broadcast_params[0] = max_logit;
    __syncthreads();
    max_logit = s_broadcast_params[0];

    float thread_sum_exp = 0.0f;
    for (int i = tid; i < V; i += blockDim.x) {
        thread_sum_exp += expf(logits[seq_idx * V + i] - max_logit);
    }
    float sum_exp = blockReduceSum(thread_sum_exp);

    if (tid == 0) {
        s_broadcast_params[0] = sum_exp;
        const float logit_label = logits[seq_idx * V + label_val];
        const float log_prob = (logit_label - max_logit) - logf(sum_exp);
        atomicAdd(total_loss, -log_prob);
    }
    __syncthreads();
    sum_exp = s_broadcast_params[0];

    const float inv_sum_exp = 1.0f / sum_exp;
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

        // Decoupled weight decay
        p -= lr * weight_decay * p;

        // Adam update
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

__global__ void embedding_backward_kernel_impl(
    float* grad_word_embeddings, const float* grad_output,
    const int* input_ids, int B, int S, int H, int V) {
    
    int word_id = blockIdx.x; // 각 블록이 하나의 단어 ID 담당
    int h_tid = threadIdx.x;  // 각 스레드가 hidden 차원 담당
    
    if (word_id < V && h_tid < H) {
        float sum = 0.0f;
        for (int i = 0; i < B * S; ++i) {
            if (input_ids[i] == word_id) {
                sum += grad_output[i * H + h_tid];
            }
        }
        grad_word_embeddings[word_id * H + h_tid] = sum;
    }
}


__global__ void accumulate_positional_embedding_grad_impl(
    float* grad_pos_embeddings, const float* grad_output,
    int B, int S, int H) {
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
    int type_id = blockIdx.x; // 0 or 1
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
        output[output_idx] = input[input_idx];
    }
}

// 개선 사항 4: 마스크 일반화 (간단한 버전 유지)
__global__ void scale_and_mask_kernel_impl(float* scores, const float* mask,
                                     int B, int N, int Sq, int Sk, float scale) {
    int b = blockIdx.x;
    int n = blockIdx.y;
    int sq = blockIdx.z;
    int sk = threadIdx.x;

    if (b < B && n < N && sq < Sq && sk < Sk) {
        int score_idx = ((b * N + n) * Sq + sq) * Sk + sk;
        int mask_idx = b * Sk + sk; // [B, 1, 1, Sk] 브로드캐스팅 가정
        scores[score_idx] = scores[score_idx] * scale + mask[mask_idx];
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
    for (int i = tid; i < N_softmax_dim; i += blockDim.x) {
        max_val = fmaxf(max_val, row_input[i]);
    }
    sdata[tid] = max_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    max_val = sdata[0];

    float sum_exp = 0.0f;
    for (int i = tid; i < N_softmax_dim; i += blockDim.x) {
        sum_exp += expf(row_input[i] - max_val);
    }
    sdata[tid] = sum_exp;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    sum_exp = sdata[0];
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
    for (int j = tid; j < N_softmax_dim; j += blockDim.x) {
        sum_val += row_grad_output[j] * row_output[j];
    }
    sdata[tid] = sum_val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    sum_val = sdata[0];

    for (int i = tid; i < N_softmax_dim; i += blockDim.x) {
        row_grad_input[i] = row_output[i] * (row_grad_output[i] - sum_val);
    }
}

__global__ void elementwise_add_kernel_impl(float* out, const float* in1, const float* in2, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in1[idx] + in2[idx];
    }
}

__global__ void accumulate_kernel_impl(float* target_and_out, const float* to_add, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        target_and_out[idx] += to_add[idx];
    }
}


// ============================================================================
// Kernel Launcher Functions
// ============================================================================
void launch_dropout_forward(
    hipStream_t stream, float* output, float* mask, const float* input,
    size_t num_elements, float prob, float scale, unsigned long long seed, unsigned long long offset)
{
    if (num_elements == 0) return;
    dim3 block_dim(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x);
    hipLaunchKernelGGL(dropout_forward_kernel_impl, grid_dim, block_dim, 0, stream,
                       output, input, mask, num_elements, prob, scale, seed, offset);
    HIP_CHECK(hipGetLastError());
}

void launch_dropout_backward(
    hipStream_t stream, float* grad_input, const float* grad_output, const float* mask,
    size_t num_elements, float scale)
{
    if (num_elements == 0) return;
    dim3 block_dim(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x);
    hipLaunchKernelGGL(dropout_backward_kernel_impl, grid_dim, block_dim, 0, stream,
                       grad_input, grad_output, mask, num_elements, scale);
    HIP_CHECK(hipGetLastError());
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
    hipLaunchKernelGGL(layer_norm_forward_kernel_warp_optimized_impl, grid, block, shared_mem_size, stream,
                       out, mean, rstd, inp, gamma, beta, B, C, epsilon);
    HIP_CHECK(hipGetLastError());
}

void launch_layer_norm_backward_optimized(
    hipStream_t stream, float* grad_input, float* grad_gamma_part, float* grad_beta_part,
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* rstd,
    int B, int C)
{
    if (B == 0 || C == 0) return;
    dim3 grid(B);
    dim3 block(std::min(C, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
    size_t shared_mem_size = ((block.x + warpSize - 1) / warpSize) * sizeof(float) * 2;
    hipLaunchKernelGGL(layer_norm_backward_kernel_optimized_impl, grid, block, shared_mem_size, stream,
                       grad_input, grad_gamma_part, grad_beta_part,
                       grad_output, input, gamma, mean, rstd, B, C);
    HIP_CHECK(hipGetLastError());
}

void launch_add_bias_gelu_kernel(hipStream_t stream, float* output, const float* input, const float* bias, int M, int N) {
    if(M==0||N==0)return;
    dim3 threads(KernelConstants::TRANSPOSE_BLOCK_DIM_X, KernelConstants::TRANSPOSE_BLOCK_DIM_Y);
    dim3 grid((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    hipLaunchKernelGGL(add_bias_gelu_kernel_impl, grid, threads, 0, stream, output, input, bias, M, N);
    HIP_CHECK(hipGetLastError());
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
    dim3 grid(N, 1); // Launch one block per feature in bias.
    size_t shared_mem_size = threads.y * sizeof(float);
    hipLaunchKernelGGL(gelu_add_bias_backward_kernel_impl, grid, threads, shared_mem_size, stream,
                       grad_input_before_bias, grad_bias_part,
                       grad_output_after_gelu, input_before_gelu,
                       M, N);
    HIP_CHECK(hipGetLastError());
}

void launch_gelu_backward_kernel(hipStream_t stream, float* grad_input, const float* grad_output, const float* input, size_t num_elements) {
    if(num_elements==0)return;
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((num_elements+block.x-1)/block.x);
    hipLaunchKernelGGL(gelu_backward_kernel_impl,grid,block,0,stream,grad_input,grad_output,input,num_elements);
    HIP_CHECK(hipGetLastError());
}

void launch_gelu_forward_kernel(hipStream_t stream, float* output, const float* input, size_t num_elements) {
    if(num_elements==0)return;
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((num_elements+block.x-1)/block.x);
    hipLaunchKernelGGL(gelu_forward_kernel_impl,grid,block,0,stream,output,input,num_elements);
    HIP_CHECK(hipGetLastError());
}

void launch_reduce_sum_kernel(hipStream_t stream, float* out_vec, const float* in_matrix, int rows, int cols) {
    if (rows == 0 || cols == 0) return;
    dim3 grid(cols);
    dim3 block(std::min(rows, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
    size_t shared_mem_size = block.x * sizeof(float);
    hipLaunchKernelGGL(reduce_sum_kernel_impl, grid, block, shared_mem_size, stream, out_vec, in_matrix, rows, cols);
    HIP_CHECK(hipGetLastError());
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
    hipLaunchKernelGGL(softmax_cross_entropy_loss_backward_kernel_optimized_impl, grid, block, shared_mem_size, stream,
                       grad_logits, logits, labels, total_loss, B, S, V, ignore_index);
    HIP_CHECK(hipGetLastError());
}

void launch_adamw_update_kernel(hipStream_t stream, float* weights, float* gradients, float* m, float* v, 
                               float lr, float beta1, float beta2, float epsilon, float weight_decay, int t, size_t n) {
    dim3 bl(KernelConstants::THREADS_PER_BLOCK_DEFAULT), gr((n+bl.x-1)/bl.x);
    hipLaunchKernelGGL(adamw_update_kernel_impl, gr, bl, 0, stream, weights, gradients, m, v, lr, beta1, beta2, epsilon, weight_decay, t, n);  // _impl 추가
    HIP_CHECK(hipGetLastError());
}

void launch_adamw_update_kernel(hipStream_t stream, float* weights, const float* gradients, float* m, float* v, 
                               float lr, float beta1, float beta2, float epsilon, float weight_decay, int t, size_t n) {
    if (n == 0) return;
    
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((n + block.x - 1) / block.x);
    
    hipLaunchKernelGGL(adamw_update_kernel_impl, grid, block, 0, stream, 
                       weights, gradients, m, v, lr, beta1, beta2, epsilon, weight_decay, t, n);
    HIP_CHECK(hipGetLastError());
}
void launch_add_bias_kernel(
    hipStream_t stream, float* output, const float* input, const float* bias, int M, int N)
{
    if (M == 0 || N == 0) return;
    dim3 threads(KernelConstants::TRANSPOSE_BLOCK_DIM_X, KernelConstants::TRANSPOSE_BLOCK_DIM_Y);
    dim3 grid((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    hipLaunchKernelGGL(add_bias_gelu_kernel_impl, grid, threads, 0, stream, output, input, bias, M, N);
    HIP_CHECK(hipGetLastError());
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
    hipLaunchKernelGGL(add_embeddings_kernel_impl, grid, block, 0, stream,
                       output, input_ids, token_type_ids, word_embeddings, position_embeddings,
                       token_type_embeddings, batch_size, seq_len, hidden_size, vocab_size, max_position_embeddings);
    HIP_CHECK(hipGetLastError());
}

void launch_embedding_backward_kernel(
    hipStream_t stream, float* grad_word_embeddings,
    const float* grad_output, const int* input_ids,
    int B, int S, int H, int V)
{
    if (B * S * H * V == 0) return;
    dim3 grid(V);
    dim3 block(std::min(H, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
    hipLaunchKernelGGL(embedding_backward_kernel_impl, grid, block, 0, stream,
                       grad_word_embeddings, grad_output, input_ids, B, S, H, V);
    HIP_CHECK(hipGetLastError());
}

void launch_accumulate_positional_embedding_grad(
    hipStream_t stream, float* grad_pos_embeddings, const float* grad_output, int B, int S, int H)
{
    if (B*S*H == 0) return;
    dim3 grid(S);
    dim3 block(std::min(H, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
    hipLaunchKernelGGL(accumulate_positional_embedding_grad_impl, grid, block, 0, stream,
                       grad_pos_embeddings, grad_output, B, S, H);
    HIP_CHECK(hipGetLastError());
}

void launch_accumulate_token_type_embedding_grad(
    hipStream_t stream, float* grad_token_type_embeddings,
    const float* grad_output, const int* token_type_ids,
    int B, int S, int H)
{
    if (B*S*H == 0) return;
    dim3 grid(2); // For token type 0 and 1
    dim3 block(std::min(H, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
    hipLaunchKernelGGL(accumulate_token_type_embedding_grad_impl, grid, block, 0, stream,
                       grad_token_type_embeddings, grad_output, token_type_ids, B, S, H);
    HIP_CHECK(hipGetLastError());
}

void launch_transpose_for_scores_kernel(
    hipStream_t stream, float* output, const float* input,
    int batch_size, int seq_len, int num_heads, int head_size)
{
    if (batch_size * seq_len * num_heads * head_size == 0) return;
    int hidden_size = num_heads * head_size;
    dim3 grid(batch_size, seq_len);
    // 개선 사항 10: 블록 크기를 warpSize 배수로 조정
    int block_dim = (hidden_size + warpSize - 1) / warpSize * warpSize;
    block_dim = std::min(block_dim, KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 block(block_dim);
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
    int block_dim = (hidden_size + warpSize - 1) / warpSize * warpSize;
    block_dim = std::min(block_dim, KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 block(block_dim);
    hipLaunchKernelGGL(transpose_back_kernel_impl, grid, block, 0, stream,
                       output, input, batch_size, seq_len, num_heads, head_size);
    HIP_CHECK(hipGetLastError());
}

void launch_scale_and_mask_kernel(
    hipStream_t stream, float* attention_scores, const float* attention_mask,
    int B, int N, int Sq, int Sk, float scale)
{
    if (B * N * Sq * Sk == 0) return;
    dim3 grid(B, N, Sq);
    dim3 block(std::min(Sk, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
    hipLaunchKernelGGL(scale_and_mask_kernel_impl, grid, block, 0, stream,
                       attention_scores, attention_mask, B, N, Sq, Sk, scale);
    HIP_CHECK(hipGetLastError());
}

void launch_softmax_kernel(
    hipStream_t stream, float* output, const float* input,
    int M_rows, int N_softmax_dim)
{
    if (M_rows == 0 || N_softmax_dim == 0) return;
    dim3 grid(M_rows);
    dim3 block(std::min(N_softmax_dim, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
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
    dim3 block(std::min(N_softmax_dim, KernelConstants::THREADS_PER_BLOCK_DEFAULT));
    size_t shared_mem_size = block.x * sizeof(float);
    hipLaunchKernelGGL(softmax_backward_kernel_impl, grid, block, shared_mem_size, stream,
                       grad_input, grad_output, output, M_rows, N_softmax_dim);
    HIP_CHECK(hipGetLastError());
}

void launch_elementwise_add_kernel(hipStream_t stream, float* out, const float* in1, const float* in2, size_t num_elements) {
    if (num_elements == 0) return;
    dim3 block_dim(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x);
    hipLaunchKernelGGL(elementwise_add_kernel_impl, grid_dim, block_dim, 0, stream, out, in1, in2, num_elements);
    HIP_CHECK(hipGetLastError());
}
void launch_scale_kernel(hipStream_t s, float* data, float scale_factor, size_t n) {
    if(n==0)return;
    dim3 bl(KernelConstants::THREADS_PER_BLOCK_DEFAULT), gr((n+bl.x-1)/bl.x);  // KernelConstants:: 추가
    hipLaunchKernelGGL(scale_kernel_impl,gr,bl,0,s,data,scale_factor,n);
    HIP_CHECK(hipGetLastError());
}
void launch_accumulate_kernel(hipStream_t stream, float* target_and_out, const float* to_add, size_t num_elements) {
    if (num_elements == 0) return;
    dim3 block_dim(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x);
    hipLaunchKernelGGL(accumulate_kernel_impl, grid_dim, block_dim, 0, stream, target_and_out, to_add, num_elements);
    HIP_CHECK(hipGetLastError());
}
// ...existing code...

// Additional kernel implementations
__global__ void set_identity_kernel_impl(float* matrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        matrix[idx] = (row == col) ? 1.0f : 0.0f;
    }
}

__global__ void elementwise_scale_kernel_impl(float* data, float scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

__global__ void elementwise_accumulate_kernel_impl(float* dst, const float* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

__global__ void add_diagonal_value_kernel_impl(float* matrix, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        matrix[idx * n + idx] += value;
    }
}

__global__ void power_kernel_impl(float* data, float power, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = powf(data[idx], power);
    }
}
__global__ void matrix_scale_columns_kernel(float* output, const float* input, const float* scales, int rows, int cols) {
    __shared__ float scale_shared[KernelConstants::TRANSPOSE_BLOCK_DIM_X]; // assuming cols <= 1024

    int tx = threadIdx.x;
    int col = blockIdx.x * blockDim.x + tx;

    // Load shared memory
    if (col < cols && threadIdx.y == 0) {
        scale_shared[tx] = scales[col];
    }
    __syncthreads();

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        output[idx] = input[idx] * scale_shared[tx];
    }
}
__global__ void transpose_kernel(float* dst, const float* src, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {
        dst[j * rows + i] = src[i * cols + j];
    }
}
// Launcher functions
void launch_set_identity_kernel(hipStream_t stream, float* matrix, int n) {
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((n * n + block.x - 1) / block.x);
    hipLaunchKernelGGL(set_identity_kernel_impl, grid, block, 0, stream, matrix, n);
    HIP_CHECK(hipGetLastError());
}

void launch_elementwise_scale_kernel(hipStream_t stream, float* data, float scale, size_t num_elements) {
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((num_elements + block.x - 1) / block.x);
    hipLaunchKernelGGL(elementwise_scale_kernel_impl, grid, block, 0, stream, data, scale, num_elements);
    HIP_CHECK(hipGetLastError());
}

void launch_elementwise_accumulate_kernel(hipStream_t stream, float* dst, const float* src, size_t num_elements) {
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((num_elements + block.x - 1) / block.x);
    hipLaunchKernelGGL(elementwise_accumulate_kernel_impl, grid, block, 0, stream, dst, src, num_elements);
    HIP_CHECK(hipGetLastError());
}

void launch_add_diagonal_value_kernel(hipStream_t stream, float* matrix, int n, float value) {
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((n + block.x - 1) / block.x);
    hipLaunchKernelGGL(add_diagonal_value_kernel_impl, grid, block, 0, stream, matrix, n, value);
    HIP_CHECK(hipGetLastError());
}

void launch_power_kernel(hipStream_t stream, float* data, float power, size_t num_elements) {
    dim3 block(KernelConstants::THREADS_PER_BLOCK_DEFAULT);
    dim3 grid((num_elements + block.x - 1) / block.x);
    hipLaunchKernelGGL(power_kernel_impl, grid, block, 0, stream, data, power, num_elements);
    HIP_CHECK(hipGetLastError());
}

void launch_matrix_scale_columns_kernel(hipStream_t stream, float* output, const float* input, const float* scales, int rows, int cols) {
    constexpr int TILE_DIM = 16;
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);

    hipLaunchKernelGGL(matrix_scale_columns_kernel, gridDim, blockDim, 0, stream, output, input, scales, rows, cols);
}


void launch_transpose_kernel(hipStream_t stream, float* dst, const float* src, int rows, int cols) {
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    hipLaunchKernelGGL(transpose_kernel, grid, block, 0, stream, dst, src, rows, cols);
}