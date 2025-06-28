#pragma once

#include "common_hip.hpp"
#include "bert_config.hpp"
#include "nn_layers_hip.hpp"
#include "attention_hip.hpp"
#include <vector>
#include <memory>

// --- Cache Structs ---

struct BertEmbeddingsCache {
    GpuTensor summed_embeddings;
    GpuTensor embeddings_output;
    LayerNormCache layernorm_cache;
    DropoutCache dropout_cache;
    const GpuTensor* input_ids_ptr = nullptr;
    const GpuTensor* token_type_ids_ptr = nullptr;
};

struct BertLayerCache {
    BertAttentionCache attention_cache;
    DenseCache ffn_intermediate_dense_cache;
    DenseCache ffn_output_dense_cache;
    DropoutCache ffn_output_dropout_cache;
    LayerNormCache ffn_output_layernorm_cache;
    const GpuTensor* layer_input_ptr = nullptr;
};

struct BertEncoderCache {
    std::vector<BertLayerCache> layer_caches;
    BertEncoderCache(int num_layers) {
        layer_caches.resize(num_layers);
    }
    BertEncoderCache() = default;
};

struct BertModelCache {
    BertEmbeddingsCache embeddings_cache;
    BertEncoderCache encoder_cache;
    BertModelCache(int num_layers) : encoder_cache(num_layers) {}
};

// --- Component Classes ---

class BertEmbeddings {
public:
    BertEmbeddings(const BertConfig& config, const std::string& name_prefix);
    std::vector<Parameter*> get_parameters();
    Parameter* get_word_embedding_params();
    void forward(hipStream_t stream,
                 const GpuTensor& input_ids, const GpuTensor& token_type_ids,
                 GpuTensor& output_embeddings, BertEmbeddingsCache& cache, bool is_training);
    void backward(hipStream_t stream,
                  const GpuTensor& grad_output_embeddings, BertEmbeddingsCache& cache);

private:
    const BertConfig& config_;
    Parameter word_embeddings_;
    Parameter position_embeddings_;
    Parameter token_type_embeddings_;
    LayerNorm layernorm_;
    Dropout dropout_;
};

class BertLayer {
public:
    BertLayer(const BertConfig& config, const std::string& name_prefix);
    std::vector<Parameter*> get_parameters();
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& hidden_states, const GpuTensor& attention_mask,
                 GpuTensor& output_states, BertLayerCache& cache, bool is_training);
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_output, BertLayerCache& cache, GpuTensor& grad_input);
private:
    const BertConfig& config_;
    BertAttention attention_;
    DenseLayer ffn_intermediate_dense_;
    DenseLayer ffn_output_dense_;
    Dropout ffn_output_dropout_;
    LayerNorm ffn_output_layernorm_;
};

class BertEncoder {
public:
    BertEncoder(const BertConfig& config, const std::string& name_prefix);
    std::vector<Parameter*> get_parameters();
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& initial_hidden_states,
                 const GpuTensor& attention_mask,
                 GpuTensor& final_hidden_states,
                 BertEncoderCache& cache,
                 bool is_training);
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_final_hidden_states,
                  BertEncoderCache& cache,
                  GpuTensor& grad_initial_hidden_states);

private:
    std::vector<std::unique_ptr<BertLayer>> layers_;
};

class BertModel {
public:
    BertModel(const BertConfig& config, const std::string& name_prefix);
    std::vector<Parameter*> get_parameters();
    Parameter* get_word_embedding_params();
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& input_ids, const GpuTensor& attention_mask,
                 const GpuTensor& token_type_ids, GpuTensor& sequence_output,
                 BertModelCache& cache, bool is_training);
    // FIXED: Corrected the function signature by adding a comma and variable name.
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_sequence_output, BertModelCache& cache);

private:
    const BertConfig& config_;
    BertEmbeddings embeddings_;
    BertEncoder encoder_;
};