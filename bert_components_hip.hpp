#ifndef BERT_COMPONENTS_HIP_HPP
#define BERT_COMPONENTS_HIP_HPP

#include "common_hip.hpp"
#include "nn_layers_hip.hpp"
#include "bert_config.hpp"
#include "attention_hip.hpp"
#include <vector>
#include <memory>

// --- Cache Structs ---
struct BertEmbeddingsCache {
    const GpuTensor* input_ids_ptr = nullptr;
    const GpuTensor* token_type_ids_ptr = nullptr;
    GpuTensor summed_embeddings;
    GpuTensor embeddings_output;
    LayerNormCache layernorm_cache;
    DropoutCache dropout_cache;
};

struct BertLayerCache {
    BertAttentionCache attention_cache;
    DenseCache ffn_intermediate_dense_cache;
    DenseCache ffn_output_dense_cache;
    DropoutCache ffn_output_dropout_cache;
    LayerNormCache ffn_output_layernorm_cache;
    
    const GpuTensor* layer_input_ptr = nullptr;
    GpuTensor attention_output;
    GpuTensor intermediate_output;
    GpuTensor ffn_output_dense_result;
    GpuTensor ffn_residual_sum_output;
};

struct BertEncoderCache {
    std::vector<BertLayerCache> layer_caches;
};

struct BertModelCache {
    BertEmbeddingsCache embeddings_cache;
    BertEncoderCache encoder_cache;
    
    BertModelCache(int num_layers) {
        encoder_cache.layer_caches.resize(num_layers);
    }
};

// --- Classes ---
class BertEmbeddings {
private:
    const BertConfig& config_;
    Parameter word_embeddings_;
    Parameter position_embeddings_;
    Parameter token_type_embeddings_;
    LayerNorm layernorm_;
    Dropout dropout_;

public:
    BertEmbeddings(const BertConfig& config, const std::string& name_prefix);
    std::vector<Parameter*> get_parameters();
    Parameter* get_word_embedding_params();
    
    void forward(hipStream_t stream,
                 const GpuTensor& input_ids, const GpuTensor& token_type_ids,
                 GpuTensor& output_embeddings, BertEmbeddingsCache& cache, bool is_training = false);
    
    void backward(hipStream_t stream,
                  const GpuTensor& grad_output_embeddings, BertEmbeddingsCache& cache);
};

class BertLayer {
private:
    const BertConfig& config_;
    BertAttention attention_;
    DenseLayer ffn_intermediate_dense_;
    DenseLayer ffn_output_dense_;
    Dropout ffn_output_dropout_;
    LayerNorm ffn_output_layernorm_;

public:
    BertLayer(const BertConfig& config, const std::string& name_prefix);
    std::vector<Parameter*> get_parameters();
    
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& input_hidden_states,
                 const GpuTensor& attention_mask,
                 GpuTensor& output_hidden_states,
                 BertLayerCache& cache,
                 bool is_training = false);
    
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_output_hidden_states,
                  BertLayerCache& cache,
                  GpuTensor& grad_input_hidden_states);
};

class BertEncoder {
private:
    std::vector<std::unique_ptr<BertLayer>> layers_;

public:
    BertEncoder(const BertConfig& config, const std::string& name_prefix);
    std::vector<Parameter*> get_parameters();
    
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& initial_hidden_states,
                 const GpuTensor& attention_mask,
                 GpuTensor& final_hidden_states,
                 BertEncoderCache& cache,
                 bool is_training = false);
    
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_final_hidden_states,
                  BertEncoderCache& cache,
                  GpuTensor& grad_initial_hidden_states);
};

class BertModel {
private:
    const BertConfig& config_;
    BertEmbeddings embeddings_;
    BertEncoder encoder_;

public:
    BertModel(const BertConfig& config, const std::string& name_prefix);
    std::vector<Parameter*> get_parameters();
    Parameter* get_word_embedding_params();
    
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& input_ids, const GpuTensor& attention_mask,
                 const GpuTensor& token_type_ids, GpuTensor& sequence_output,
                 BertModelCache& cache, bool is_training = false);
    
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_sequence_output, BertModelCache& cache);
};

#endif // BERT_COMPONENTS_HIP_HPP