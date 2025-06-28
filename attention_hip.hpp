#ifndef ATTENTION_HIP_HPP
#define ATTENTION_HIP_HPP

#include "common_hip.hpp"
#include "bert_config.hpp"
#include "nn_layers_hip.hpp" // DenseLayer, Dropout, LayerNorm 등이 여기에 있음

// --- Forward Declarations ---
// 이곳에는 전방 선언이 필요 없습니다.

// --- Cache Structs ---
struct SelfAttentionCache {
    GpuTensor q_proj, k_proj, v_proj;
    GpuTensor q_reshaped, k_reshaped, v_reshaped;
    GpuTensor attention_scores, attention_probs;
    GpuTensor context_reshaped, context_layer;

    DenseCache q_dense_cache, k_dense_cache, v_dense_cache;
    DropoutCache attention_probs_dropout_cache;
    const GpuTensor* input_hidden_states = nullptr;
};

struct BertAttentionCache {
    SelfAttentionCache self_attention_cache;
    DenseCache output_dense_cache;
    DropoutCache output_dropout_cache;
    LayerNormCache output_layernorm_cache;
    const GpuTensor* attention_input = nullptr;
};

// --- Class Definitions ---
class BertSelfAttention {
public:
    BertSelfAttention(const BertConfig& config, const std::string& name_prefix);
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& hidden_states, const GpuTensor& attention_mask,
                 SelfAttentionCache& cache, bool is_training = false);
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_context_layer_output,
                  SelfAttentionCache& cache,
                  GpuTensor& grad_input_hidden_states);
    std::vector<Parameter*> get_parameters();

private:
    DenseLayer query_;
    DenseLayer key_;
    DenseLayer value_;
    Dropout dropout_;
    int num_attention_heads_;
    int attention_head_size_;
};

class BertAttention {
public:
    BertAttention(const BertConfig& config, const std::string& name_prefix);
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& input_tensor,
                 const GpuTensor& attention_mask,
                 GpuTensor& output_tensor,
                 BertAttentionCache& cache,
                 bool is_training = false);
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_final_output,
                  BertAttentionCache& cache,
                  GpuTensor& grad_input_tensor);
    std::vector<Parameter*> get_parameters();

private:
    BertSelfAttention self_attention_;
    DenseLayer output_dense_;
    Dropout output_dropout_;
    LayerNorm output_layernorm_;
};

#endif // ATTENTION_HIP_HPP