#ifndef LANGUAGE_MODEL_HIP_HPP
#define LANGUAGE_MODEL_HIP_HPP

#include "common_hip.hpp"
#include "nn_layers_hip.hpp"        
#include "bert_components_hip.hpp"  
#include "optimizer_hip.hpp"
#include <memory>

// --- Cache Struct for BertLMPredictionHead ---
struct BertLMPredictionHeadCache {
    // Input to this head (usually sequence_output from BertModel)
    const GpuTensor* hidden_states_input = nullptr;

    // Intermediate tensors for the transform part
    GpuTensor transform_dense_output;
    GpuTensor transform_gelu_output; 
    GpuTensor transform_layernorm_output; 

    // Caches for sub-modules (nn_layers_hip.hpp의 이름과 일치)
    DenseCache transform_dense_cache;          // DenseLayerCache → DenseCache
    GeluCache transform_gelu_cache; 
    LayerNormCache transform_layernorm_cache;
    
    void clear() {
        hidden_states_input = nullptr;
        transform_dense_output.free();
        transform_gelu_output.free();
        transform_layernorm_output.free();
        // Cache들에는 clear() 메서드가 없으므로 생략
    }
};

// --- BertLMPredictionHead 클래스 ---
class BertLMPredictionHead {
private:
    const BertConfig& config_;
    DenseLayer transform_dense_;
    Gelu transform_gelu_;
    LayerNorm transform_layernorm_;
    Parameter& shared_word_embeddings_;
    Parameter decoder_bias_;

public:
    BertLMPredictionHead(const BertConfig& config, Parameter& shared_word_embeddings_parameter, const std::string& name_prefix = "cls.predictions");
    
    std::vector<Parameter*> get_parameters();
    
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& hidden_states, GpuTensor& logits, 
                 BertLMPredictionHeadCache& cache);
    
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_logits, BertLMPredictionHeadCache& cache,
                  GpuTensor& grad_hidden_states);
};

// --- CANBertForMaskedLM 클래스 ---
class CANBertForMaskedLM {
private:
    BertConfig config_;
    std::unique_ptr<BertModel> bert_model_;
    std::unique_ptr<BertLMPredictionHead> lm_head_;
    std::unique_ptr<AdamWOptimizer> optimizer_;
    rocblas_handle blas_handle_;
    hipStream_t stream_;

public:
    CANBertForMaskedLM(const BertConfig& cfg);
    ~CANBertForMaskedLM();
    
    void initialize_parameters(float mean = 0.0f, float stddev = 0.02f);
    std::vector<Parameter*> get_parameters();
    
    void train_step(const GpuTensor& input_ids,
                    const GpuTensor& attention_mask,
                    const GpuTensor& token_type_ids,
                    const GpuTensor& labels,
                    GpuTensor& loss);
};

#endif // LANGUAGE_MODEL_HIP_HPP