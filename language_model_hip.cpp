#include "language_model_hip.hpp"
#include "common_hip.hpp"
#include "hip_kernels.hpp" // For loss kernel, reduction kernels
#include "bert_components_hip.hpp" // For BertConfig, BertModel parameter access
#include <stdexcept>
#include <vector>
#include <string>

// ============================================================================
// BertLMPredictionHead Implementation
// ============================================================================
BertLMPredictionHead::BertLMPredictionHead(const BertConfig& config, Parameter& shared_word_embeddings_parameter, const std::string& name_prefix)
    : config_(config),  // config_ 먼저 초기화
      transform_dense_(config, config.hidden_size, config.hidden_size, name_prefix + ".transform.dense"),
      transform_layernorm_(config.hidden_size, config.layer_norm_eps, name_prefix + ".transform.LayerNorm"),
      shared_word_embeddings_(shared_word_embeddings_parameter),
      decoder_bias_({config.vocab_size}, name_prefix + ".bias") {}  // bias_ → decoder_bias_

std::vector<Parameter*> BertLMPredictionHead::get_parameters() {
    auto dense_params = transform_dense_.get_parameters();
    auto ln_params = transform_layernorm_.get_parameters();
    
    std::vector<Parameter*> params;
    params.insert(params.end(), dense_params.begin(), dense_params.end());
    params.insert(params.end(), ln_params.begin(), ln_params.end());
    params.push_back(&decoder_bias_);  // bias_ → decoder_bias_
    return params;
}

void BertLMPredictionHead::forward(rocblas_handle blas_handle, hipStream_t stream,
                                   const GpuTensor& hidden_states, GpuTensor& logits, 
                                   BertLMPredictionHeadCache& cache) {
    int batch_size = hidden_states.dim_size(0);
    int seq_len = hidden_states.dim_size(1);
    std::vector<int> logits_dims = {batch_size, seq_len, config_.vocab_size};
    
    if (!logits.is_allocated() || logits.dims_ != logits_dims) {
        logits.allocate(logits_dims);
    }
    
    cache.hidden_states_input = &hidden_states;

    // Dense layer forward
    cache.transform_dense_output.allocate(hidden_states.dims_);
    transform_dense_.forward(blas_handle, stream, hidden_states, cache.transform_dense_output, cache.transform_dense_cache);

    // GELU forward  
    cache.transform_gelu_output.allocate(cache.transform_dense_output.dims_);
    transform_gelu_.forward(stream, cache.transform_dense_output, cache.transform_gelu_output, cache.transform_gelu_cache);

    // LayerNorm forward
    cache.transform_layernorm_output.allocate(cache.transform_gelu_output.dims_);
    transform_layernorm_.forward(stream, cache.transform_gelu_output, cache.transform_layernorm_output, cache.transform_layernorm_cache);

    // Matrix multiplication with shared word embeddings
    int M = batch_size * seq_len;
    int K = config_.hidden_size;
    int N = config_.vocab_size;
    
    const float alpha = 1.0f, beta = 0.0f;
    ROCBLAS_CHECK(rocblas_sgemm(blas_handle, rocblas_operation_none, rocblas_operation_transpose,
                                M, N, K, &alpha,
                                (const float*)cache.transform_layernorm_output.d_ptr_, K,
                                (const float*)shared_word_embeddings_.weights.d_ptr_, K,
                                &beta,
                                (float*)logits.d_ptr_, N));

    // Add decoder bias
    launch_add_bias_kernel(stream,
                           (float*)logits.d_ptr_,
                           (const float*)logits.d_ptr_,
                           (const float*)decoder_bias_.weights.d_ptr_,
                           M, N);
}

void BertLMPredictionHead::backward(rocblas_handle blas_handle, hipStream_t stream,
                                    const GpuTensor& grad_logits, BertLMPredictionHeadCache& cache,
                                    GpuTensor& grad_hidden_states) {
    if (!grad_logits.is_allocated() || !cache.hidden_states_input || !cache.hidden_states_input->is_allocated() ||
        !cache.transform_layernorm_output.is_allocated()) {
        throw std::runtime_error("Required tensors not allocated for BertLMPredictionHead::backward");
    }

    if(!grad_hidden_states.is_allocated() || grad_hidden_states.dims_ != cache.hidden_states_input->dims_){
        grad_hidden_states.allocate(cache.hidden_states_input->dims_);
    }

    // Allocate gradients for all parameters - public 메서드 사용
    transform_dense_.allocate_gradients();
    transform_layernorm_.allocate_gradients();
    decoder_bias_.allocate_gradients();

    int batch_size = cache.hidden_states_input->dim_size(0);
    int seq_len = cache.hidden_states_input->dim_size(1);
    int vocab_size = config_.vocab_size;
    int K_hidden = cache.hidden_states_input->dim_size(2);

    // Step 1: Compute grad_bias (reduce sum over batch and sequence dimensions)
    launch_reduce_sum_kernel(stream,
                            (float*)decoder_bias_.grad_weights.d_ptr_,
                            (const float*)grad_logits.d_ptr_,
                            batch_size * seq_len, vocab_size);

    // Step 2: Compute gradients w.r.t. shared word embeddings
    int M_vocab = vocab_size;
    int N_hidden = K_hidden;
    int K_batch = batch_size * seq_len;

    const float alpha = 1.0f, beta = 1.0f; // Use beta=1 for accumulating gradients

    ROCBLAS_CHECK(rocblas_sgemm(blas_handle, rocblas_operation_transpose, rocblas_operation_none,
                                M_vocab, N_hidden, K_batch, &alpha,
                                (const float*)grad_logits.d_ptr_, vocab_size,
                                (const float*)cache.transform_layernorm_output.d_ptr_, K_hidden,
                                &beta,
                                (float*)shared_word_embeddings_.grad_weights.d_ptr_, K_hidden));

    // Step 3: Compute gradients w.r.t. transform_layernorm_output  
    GpuTensor grad_transform_layernorm_output;
    grad_transform_layernorm_output.allocate(cache.transform_layernorm_output.dims_);

    ROCBLAS_CHECK(rocblas_sgemm(blas_handle, rocblas_operation_none, rocblas_operation_none,
                                K_batch, N_hidden, M_vocab, &alpha,
                                (const float*)grad_logits.d_ptr_, vocab_size,
                                (const float*)shared_word_embeddings_.weights.d_ptr_, K_hidden,
                                &beta,
                                (float*)grad_transform_layernorm_output.d_ptr_, K_hidden));

    // Step 4: Backward through LayerNorm
    GpuTensor grad_transform_gelu_output;
    grad_transform_gelu_output.allocate(cache.transform_gelu_output.dims_);
    transform_layernorm_.backward(stream, grad_transform_layernorm_output, cache.transform_layernorm_cache, grad_transform_gelu_output);

    // Step 5: Backward through GELU
    GpuTensor grad_transform_dense_output;
    grad_transform_dense_output.allocate(cache.transform_dense_output.dims_);
    transform_gelu_.backward(stream, grad_transform_gelu_output, grad_transform_dense_output, cache.transform_gelu_cache);

    // Step 6: Backward through Dense layer
    transform_dense_.backward(blas_handle, stream, grad_transform_dense_output, cache.transform_dense_cache, grad_hidden_states);
}

// ============================================================================
// CANBertForMaskedLM Implementation
// ============================================================================
CANBertForMaskedLM::CANBertForMaskedLM(const BertConfig& cfg) : config_(cfg) { // Make a copy of config or ensure lifetime
    bert_model_ = std::make_unique<BertModel>(config_, "bert"); // Pass config by const ref
    // Pass the actual Parameter object for word embeddings from bert_model_ to lm_head_
    lm_head_ = std::make_unique<BertLMPredictionHead>(config_, *bert_model_->get_word_embedding_params(), "cls.predictions");

    ROCBLAS_CHECK(rocblas_create_handle(&blas_handle_));
    HIP_CHECK(hipStreamCreate(&stream_)); // Using default stream 0 can also be an option if no specific stream is needed.

    // Optimizer setup
    auto model_params = get_parameters(); // Collect all parameters
    for(auto* p : model_params) {
        if (p) p->allocate_gradients(); // Ensure gradients are allocated for all params
    }
    // Default AdamW params from original user code. Configurable if needed.
    optimizer_ = std::make_unique<AdamWOptimizer>(model_params, 1e-4f, 0.9f, 0.999f, 1e-6f, 0.01f);
}

CANBertForMaskedLM::~CANBertForMaskedLM() {
    if (stream_) {
        hipError_t err = hipStreamDestroy(stream_);
        if (err != hipSuccess) {
             fprintf(stderr, "HIP Error: Failed to destroy stream in CANBertForMaskedLM destructor: %s\n", hipGetErrorString(err));
        }
    }
    if (blas_handle_) rocblas_destroy_handle(blas_handle_);
}

void CANBertForMaskedLM::initialize_parameters(float mean, float stddev) {
    for (auto* p : get_parameters()) {
        if (p) p->initialize_random(mean, stddev);
    }
}

std::vector<Parameter*> CANBertForMaskedLM::get_parameters() {
    auto bert_params = bert_model_->get_parameters();
    auto head_params = lm_head_->get_parameters();
    bert_params.insert(bert_params.end(), head_params.begin(), head_params.end());
    return bert_params;
}

void CANBertForMaskedLM::train_step(const GpuTensor& input_ids,
                                    const GpuTensor& attention_mask,
                                    const GpuTensor& token_type_ids,
                                    const GpuTensor& labels,
                                    GpuTensor& loss) { // Scalar output
    if (!input_ids.is_allocated() || !attention_mask.is_allocated() || !labels.is_allocated()) {
        throw std::runtime_error("Input tensors (input_ids, attention_mask, labels) must be allocated for train_step.");
    }
    // token_type_ids is optional, BertEmbeddings handles unallocated case by creating a dummy.

     optimizer_->zero_grad(stream_); // Zero out all parameter gradients

    // --- Forward Pass ---
    BertModelCache model_cache(config_.num_hidden_layers); // num_hidden_layers from config
    BertLMPredictionHeadCache head_cache;

    GpuTensor sequence_output; // (B, S, H) - Output from BertModel
    // sequence_output needs to be allocated based on input_ids dims and config.hidden_size
    sequence_output.allocate({input_ids.dim_size(0), input_ids.dim_size(1), config_.hidden_size});

    bert_model_->forward(blas_handle_, stream_, input_ids, attention_mask, token_type_ids,
                         sequence_output, model_cache, true /*is_training*/);

    GpuTensor logits; // (B, S, V) - Output from LM Head
    logits.allocate({input_ids.dim_size(0), input_ids.dim_size(1), config_.vocab_size});
    lm_head_->forward(blas_handle_, stream_, sequence_output, logits, head_cache);

    // --- Loss Calculation & Initial Gradient for Backward Pass ---
    GpuTensor grad_logits; // (B, S, V) - Gradient of loss w.r.t. logits
    grad_logits.allocate(logits.dims_);

 if (!loss.is_allocated() || loss.num_elements_ != 1) {
        loss.allocate({1}); // Scalar loss
    }
    loss.zero_out(stream_); // Initialize loss to zero for accumulation

    // This kernel calculates dL/dLogits and sum(-log_probs) for the loss.
   launch_softmax_cross_entropy_loss_backward_optimized(stream_,
                                               (float*)grad_logits.d_ptr_,
                                               (const float*)logits.d_ptr_,
                                               (const int*)labels.d_ptr_,
                                               (float*)loss.d_ptr_,
                                               input_ids.dim_size(0), // B
                                               input_ids.dim_size(1), // S
                                               config_.vocab_size,    // V
                                               -100);      
    HIP_CHECK(hipGetLastError()); // Check after kernel launch

    // --- Backward Pass ---
    GpuTensor grad_sequence_output; // (B, S, H) - Grad w.r.t. output of BertModel
    grad_sequence_output.allocate(sequence_output.dims_);

    lm_head_->backward(blas_handle_, stream_, grad_logits, head_cache, grad_sequence_output);
    bert_model_->backward(blas_handle_, stream_, grad_sequence_output, model_cache);

    // --- Optimizer Step ---
    optimizer_->step(stream_);

    // Synchronize stream to ensure all operations are complete, especially if loss is read by CPU next.
    HIP_CHECK(hipStreamSynchronize(stream_));
}
