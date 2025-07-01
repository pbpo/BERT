#include "bert_components_hip.hpp"
#include "common_hip.hpp"
#include "hip_kernels.hpp" // 실제 커널 런처 함수들을 포함합니다.
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>
// ============================================================================
// BertEmbeddings 구현부
// ============================================================================

BertEmbeddings::BertEmbeddings(const BertConfig& config, const std::string& name_prefix)
    : config_(config),
      word_embeddings_({config.vocab_size, config.hidden_size}, name_prefix + ".word_embeddings"),
      position_embeddings_({config.max_position_embeddings, config.hidden_size}, name_prefix + ".position_embeddings"),
      token_type_embeddings_({2, config.hidden_size}, name_prefix + ".token_type_embeddings"),
      layernorm_(config.hidden_size, config.layer_norm_eps, name_prefix + ".LayerNorm"),
      dropout_(config.hidden_dropout_prob) {}

std::vector<Parameter*> BertEmbeddings::get_parameters() {
    auto ln_params = layernorm_.get_parameters();
    std::vector<Parameter*> params = {&word_embeddings_, &position_embeddings_, &token_type_embeddings_};
    params.insert(params.end(), ln_params.begin(), ln_params.end());
    return params;
}

Parameter* BertEmbeddings::get_word_embedding_params() {
    return &word_embeddings_;
}

void BertEmbeddings::forward(hipStream_t stream,
                             const GpuTensor& input_ids,
                             const GpuTensor& token_type_ids,
                             GpuTensor& output_embeddings,
                             BertEmbeddingsCache& cache,
                             bool is_training) {
    std::cout << "[BertEmbeddings::forward] 시작" << std::endl;
    if (!input_ids.is_allocated() || !word_embeddings_.weights.is_allocated() ||
        !position_embeddings_.weights.is_allocated() || !token_type_embeddings_.weights.is_allocated()) {
        throw std::runtime_error("Required tensors for BertEmbeddings::forward are not allocated.");
    }
    const int batch_size = input_ids.dim_size(0);
    const int seq_len    = input_ids.dim_size(1);
    std::vector<int> expected_output_dims = {batch_size, seq_len, config_.hidden_size};

    std::cout << "  [STEP] output_embeddings.allocate(" << batch_size << ", " << seq_len << ", " << config_.hidden_size << ")" << std::endl;
    output_embeddings.allocate(expected_output_dims);

    std::cout << "  [STEP] cache.summed_embeddings.allocate(...)" << std::endl;
    cache.summed_embeddings.allocate(expected_output_dims);
    std::cout << "  [STEP] cache.embeddings_output.allocate(...)" << std::endl;
    cache.embeddings_output.allocate(expected_output_dims);
    cache.input_ids_ptr = &input_ids;


     const GpuTensor* token_type_ids_to_use = &token_type_ids;
    if (!token_type_ids.is_allocated()) {
        // cache 내의 storage를 사용
        cache.dummy_token_type_ids_storage.dtype = DataType::INT32;
        cache.dummy_token_type_ids_storage.allocate(input_ids.dims_);
        cache.dummy_token_type_ids_storage.zero_out(stream);
        token_type_ids_to_use = &cache.dummy_token_type_ids_storage; // cache 내부 텐서의 주소를 가리킴
    }
    cache.token_type_ids_ptr = token_type_ids_to_use;

    std::cout << "  [STEP] launch_add_embeddings_kernel 실행" << std::endl;
    launch_add_embeddings_kernel(
        stream,
        (float*)cache.summed_embeddings.d_ptr_,
        (const int*)input_ids.d_ptr_, (const int*)token_type_ids_to_use->d_ptr_,
        (const float*)word_embeddings_.weights.d_ptr_,
        (const float*)position_embeddings_.weights.d_ptr_,
        (const float*)token_type_embeddings_.weights.d_ptr_,
        batch_size, seq_len, config_.hidden_size,
        config_.vocab_size, config_.max_position_embeddings);
        HIP_CHECK( hipGetLastError() ); 
    std::cout << "    [LOG] 임베딩 합산 완료" << std::endl;

    std::cout << "  [STEP] layernorm_.forward 실행" << std::endl;
    cache.summed_embeddings.copy_from_gpu(cache.summed_embeddings, stream);
    layernorm_.forward(stream, cache.summed_embeddings, cache.embeddings_output, cache.layernorm_cache);
    std::cout << "    [LOG] LayerNorm 완료" << std::endl;

    std::cout << "  [STEP] dropout_.forward 실행" << std::endl;
    dropout_.forward(stream, cache.embeddings_output, cache.dropout_cache, is_training);
    std::cout << "    [LOG] Dropout 완료 (is_training=" << is_training << ")" << std::endl;

    std::cout << "  [STEP] output_embeddings.copy_from_gpu 실행" << std::endl;
    output_embeddings.copy_from_gpu(cache.embeddings_output, stream);
    std::cout << "[BertEmbeddings::forward] 종료" << std::endl;
}

void BertEmbeddings::backward(hipStream_t stream,
                              const GpuTensor& grad_output_embeddings,
                              BertEmbeddingsCache& cache) {
    std::cout << "[BertEmbeddings::backward] 시작" << std::endl;
    if (!grad_output_embeddings.is_allocated() ||
        !cache.input_ids_ptr || !cache.summed_embeddings.is_allocated()) {
        throw std::runtime_error("Required tensors/cache not ready for BertEmbeddings::backward.");
    }

    std::cout << "  [STEP] word_embeddings_.allocate_gradients()" << std::endl;
    word_embeddings_.allocate_gradients();
    std::cout << "  [STEP] position_embeddings_.allocate_gradients()" << std::endl;
    position_embeddings_.allocate_gradients();
    std::cout << "  [STEP] token_type_embeddings_.allocate_gradients()" << std::endl;
    token_type_embeddings_.allocate_gradients();

    GpuTensor grad_after_dropout, grad_after_layernorm;
    std::cout << "  [STEP] grad_after_dropout.allocate(...)" << std::endl;
    grad_after_dropout.allocate(grad_output_embeddings.dims_);
    std::cout << "  [STEP] grad_after_layernorm.allocate(...)" << std::endl;
    grad_after_layernorm.allocate(grad_output_embeddings.dims_);

    std::cout << "  [STEP] grad_after_dropout.copy_from_gpu 실행" << std::endl;
    grad_after_dropout.copy_from_gpu(grad_output_embeddings, stream);

    std::cout << "  [STEP] dropout_.backward 실행" << std::endl;
        dropout_.backward(stream, 
                      grad_after_dropout, 
                      grad_after_dropout, 
                      cache.dropout_cache); // 4개 매개변수 버전
    std::cout << "    [LOG] Dropout backward 완료" << std::endl;

    std::cout << "  [STEP] layernorm_.backward 실행" << std::endl;
    layernorm_.backward(stream, grad_after_dropout, cache.summed_embeddings, cache.layernorm_cache, grad_after_layernorm);
    std::cout << "    [LOG] LayerNorm backward 완료" << std::endl;

    const int batch_size  = cache.input_ids_ptr->dim_size(0);
    const int seq_len     = cache.input_ids_ptr->dim_size(1);
    const int hidden_size = config_.hidden_size;
    const int vocab_size  = config_.vocab_size;

    std::cout << "  [STEP] launch_embedding_backward_kernel 실행" << std::endl;
    launch_embedding_backward_kernel(
        stream,
        (float*)word_embeddings_.grad_weights.d_ptr_,
        (const float*)grad_after_layernorm.d_ptr_,
        (const int*)cache.input_ids_ptr->d_ptr_,
        batch_size, seq_len, hidden_size, vocab_size);
    std::cout << "    [LOG] Word embedding gradient 계산 완료" << std::endl;

    std::cout << "  [STEP] launch_accumulate_positional_embedding_grad 실행" << std::endl;
    launch_accumulate_positional_embedding_grad(
        stream,
        (float*)position_embeddings_.grad_weights.d_ptr_,
        (const float*)grad_after_layernorm.d_ptr_,
        batch_size, seq_len, hidden_size);
    std::cout << "    [LOG] Position embedding gradient 계산 완료" << std::endl;

    std::cout << "  [STEP] launch_accumulate_token_type_embedding_grad 실행" << std::endl;
    launch_accumulate_token_type_embedding_grad(
        stream,
        (float*)token_type_embeddings_.grad_weights.d_ptr_,
        (const float*)grad_after_layernorm.d_ptr_,
        (const int*)cache.token_type_ids_ptr->d_ptr_,
        batch_size, seq_len, hidden_size);
    std::cout << "    [LOG] Token-type embedding gradient 계산 완료" << std::endl;

    std::cout << "[BertEmbeddings::backward] 종료" << std::endl;
}

// ============================================================================
// BertLayer 구현부
// ============================================================================

BertLayer::BertLayer(const BertConfig& config, const std::string& name_prefix)
    : config_(config), // ADDED
      attention_(config, name_prefix + ".attention"),
      // FIXED: DenseLayer constructor argument order
      ffn_intermediate_dense_(config, config.hidden_size, config.intermediate_size, name_prefix + ".intermediate.dense"),
      ffn_output_dense_(config, config.intermediate_size, config.hidden_size, name_prefix + ".output.dense"),
      ffn_output_dropout_(config.hidden_dropout_prob),
      ffn_output_layernorm_(config.hidden_size, config.layer_norm_eps, name_prefix + ".output.LayerNorm")
      {}

std::vector<Parameter*> BertLayer::get_parameters() {
    auto attention_params = attention_.get_parameters();
    auto ffn_int_params = ffn_intermediate_dense_.get_parameters();
    auto ffn_out_params = ffn_output_dense_.get_parameters();
    auto ffn_ln_params = ffn_output_layernorm_.get_parameters();
    std::vector<Parameter*> params;
    params.insert(params.end(), attention_params.begin(), attention_params.end());
    params.insert(params.end(), ffn_int_params.begin(), ffn_int_params.end());
    params.insert(params.end(), ffn_out_params.begin(), ffn_out_params.end());
    params.insert(params.end(), ffn_ln_params.begin(), ffn_ln_params.end());
    return params;
}

// bert_components_hip.cpp

void BertLayer::forward(rocblas_handle blas_handle,
                        hipStream_t stream,
                        const GpuTensor& input_hidden_states,
                        const GpuTensor& attention_mask,
                        GpuTensor& output_hidden_states, // 외부에서 할당된 최종 출력 버퍼
                        BertLayerCache& cache,
                        bool is_training) 
{
    if (!input_hidden_states.is_allocated()) {
        throw std::runtime_error("Input hidden_states not allocated for BertLayer forward.");
    }
    // [REMOVED] output_hidden_states.allocate(...) 호출을 완전히 제거합니다.

    std::cout << "[BertLayer::forward] 시작 (Layer Input Ptr: " << input_hidden_states.d_ptr_ << ")" << std::endl;

    cache.layer_input_ptr = &input_hidden_states;

    // 1. Self-Attention
    // [MODIFIED] 로컬 변수 대신 캐시의 텐서에 할당하고 사용합니다.
    cache.attention_output.allocate(input_hidden_states.dims_);
    attention_.forward(blas_handle, stream,
                       input_hidden_states, attention_mask,
                       cache.attention_output, cache.attention_cache, is_training);
    std::cout << "    [LOG] Self-Attention 완료" << std::endl;

    // 2. FFN Intermediate
    // [MODIFIED] 로컬 변수 대신 캐시의 텐서에 할당하고 사용합니다.
    cache.intermediate_output.allocate({ input_hidden_states.dim_size(0),
                                         input_hidden_states.dim_size(1),
                                         config_.intermediate_size });
    ffn_intermediate_dense_.forward(blas_handle, stream,
                                    cache.attention_output, cache.intermediate_output,
                                    cache.ffn_intermediate_dense_cache);
    std::cout << "    [LOG] FFN Intermediate 완료" << std::endl;

    // 3. FFN Output + Dropout
    // [MODIFIED] 로컬 변수 대신 캐시의 텐서에 할당하고 사용합니다.
    cache.ffn_output_dense_result.allocate(cache.attention_output.dims_);
    ffn_output_dense_.forward(blas_handle, stream,
                              cache.intermediate_output, cache.ffn_output_dense_result,
                              cache.ffn_output_dense_cache);
    ffn_output_dropout_.forward(stream,
                                cache.ffn_output_dense_result,
                                cache.ffn_output_dropout_cache,
                                is_training);
    std::cout << "    [LOG] FFN Output 및 Dropout 완료" << std::endl;

    // 4. Residual Connection + Final LayerNorm
    // [MODIFIED] 로컬 변수 대신 캐시의 텐서에 할당하고 사용합니다.
    cache.ffn_residual_sum_output.allocate(cache.ffn_output_dense_result.dims_);
    launch_elementwise_add_kernel(
        stream,
        (float*)cache.ffn_residual_sum_output.d_ptr_,
        (const float*)cache.ffn_output_dense_result.d_ptr_,
        (const float*)cache.attention_output.d_ptr_, // Attention 출력과 더해야 함
        cache.ffn_residual_sum_output.num_elements_);
       
    ffn_output_layernorm_.forward(
        stream,
        cache.ffn_residual_sum_output,
        output_hidden_states, // 최종 결과를 외부 버퍼에 씀
        cache.ffn_output_layernorm_cache);
    std::cout << "    [LOG] Residual 합산 및 LayerNorm 완료" << std::endl;

    std::cout << "[BertLayer::forward] 종료" << std::endl;
}

void BertLayer::backward(rocblas_handle blas_handle, hipStream_t stream,
                         const GpuTensor& grad_output_hidden_states,
                         BertLayerCache& cache, // [수정] const 제거
                         GpuTensor& grad_input_hidden_states) {

    printf("[BertLayer::backward] 시작 (Grad Output Ptr: %p)\n", grad_output_hidden_states.d_ptr_);

cache.grad_ffn_sum_input.allocate(grad_output_hidden_states.dims_);
ffn_output_layernorm_.backward(stream, grad_output_hidden_states, cache.ffn_residual_sum_output, cache.ffn_output_layernorm_cache, cache.grad_ffn_sum_input);

// ======================= [디버깅 코드 블록] =======================
// 헬퍼 람다 함수 (GPU 텐서의 앞 5개 원소를 출력)
auto print_tensor_head = [&](const std::string& name, const GpuTensor& tensor) {
    if (!tensor.is_allocated()) {
        printf("  [DEBUG] %s: Not Allocated\n", name.c_str());
        return;
    }
    printf("  [DEBUG] %s (ptr: %p)\n", name.c_str(), tensor.d_ptr_);
    std::vector<float> host_data(10); // Print first 10
    HIP_CHECK(hipMemcpy(host_data.data(), tensor.d_ptr_, 10 * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < 5; ++i) {
        printf("    > %s[%d] = %f\n", name.c_str(), i, host_data[i]);
    }
};

printf("\n--- FFN 역전파 단계별 값 추적 ---\n");
HIP_CHECK(hipStreamSynchronize(stream)); // 이전 작업 완료 대기
printf("--- 1. LayerNorm 역전파 직후 ---\n");
print_tensor_head("grad_ffn_sum_input", cache.grad_ffn_sum_input);
// =================================================================

// --- 2. FFN 잔차 연결(Residual Connection) 역전파 ---
cache.grad_ffn_dropout_output.allocate(cache.grad_ffn_sum_input.dims_);
cache.grad_ffn_dropout_output.copy_from_gpu(cache.grad_ffn_sum_input, stream);

cache.grad_attention_block_output.allocate(cache.attention_output.dims_);
cache.grad_attention_block_output.copy_from_gpu(cache.grad_ffn_sum_input, stream);

// --- 3. FFN 블록 역전파 (Dropout -> Output Dense -> Intermediate Dense) ---
cache.grad_intermediate_output.allocate(cache.intermediate_output.dims_);
cache.grad_ffn_input.allocate(cache.attention_output.dims_);

ffn_output_dropout_.backward(stream,
                             cache.grad_ffn_dropout_output,
                             cache.grad_ffn_dropout_output,
                             cache.ffn_output_dropout_cache);

// ======================= [디버깅 코드 블록] =======================
//HIP_CHECK(hipStreamSynchronize(stream));
//printf("--- 2. Dropout 역전파 직후 ---\n");
//print_tensor_head("grad_ffn_dropout_output", cache.grad_ffn_dropout_output);
// =================================================================

ffn_output_dense_.backward(blas_handle, stream, cache.grad_ffn_dropout_output, cache.ffn_output_dense_cache, cache.grad_intermediate_output);

// ======================= [디버깅 코드 블록] =======================
//HIP_CHECK(hipStreamSynchronize(stream));
//printf("--- 3. Output Dense 역전파 직후 ---\n");
//print_tensor_head("grad_intermediate_output", cache.grad_intermediate_output);
// =================================================================

ffn_intermediate_dense_.backward(blas_handle, stream, cache.grad_intermediate_output, cache.ffn_intermediate_dense_cache, cache.grad_ffn_input);

// ======================= [디버깅 코드 블록] =======================
//HIP_CHECK(hipStreamSynchronize(stream));
//printf("--- 4. Intermediate Dense 역전파 직후 (FFN 최종 결과) ---\n");
//print_tensor_head("grad_ffn_input", cache.grad_ffn_input);
//printf("----------------------------------\n\n");
// =================================================================

  //  printf("    [DEBUG] 동기화 시작 (커널 결과 확인 전)\n");
   // HIP_CHECK(hipStreamSynchronize(stream));
    //printf("    [DEBUG] 동기화 완료\n");


    //printf("    [LOG] FFN 블록 전체 역전파 완료\n");
    //printf("    --- accumulate_kernel 입력 값 검사 ---\n");
    //print_tensor_head("target (grad_attn_out)", cache.grad_attention_block_output);
    //print_tensor_head("to_add (grad_ffn_in)", cache.grad_ffn_input);
    //printf("    -------------------------------------\n");

    // --- 4. 어텐션 블록의 잔차 연결 그래디언트 합산 ---
    // 어텐션 블록의 최종 그래디언트 = (FFN을 통과한 그래디언트) + (잔차 연결에서 직접 온 그래디언트)
    launch_accumulate_kernel(stream,
                             (float*)cache.grad_attention_block_output.d_ptr_,
                             (const float*)cache.grad_ffn_input.d_ptr_,
                             cache.grad_attention_block_output.num_elements_);
    printf("    [LOG] 어텐션 블록 그래디언트 합산 완료\n");

    // --- 5. 어텐션 블록 역전파 ---
    // 최종적으로 합산된 그래디언트를 어텐션 블록으로 전달합니다.
    attention_.backward(blas_handle, stream, cache.grad_attention_block_output, cache.attention_cache, grad_input_hidden_states);
}



// ============================================================================
// BertEncoder & BertModel 구현부 (이전과 동일, 완전한 상태)
// ============================================================================

BertEncoder::BertEncoder(const BertConfig& config, const std::string& name_prefix) {
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(std::make_unique<BertLayer>(config, name_prefix + ".layer." + std::to_string(i)));
    }
}

std::vector<Parameter*> BertEncoder::get_parameters() {
    std::vector<Parameter*> params;
    for (const auto& layer : layers_) {
        auto layer_params = layer->get_parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

void BertEncoder::forward(rocblas_handle blas_handle, hipStream_t stream,
                          const GpuTensor& initial_hidden_states,
                          const GpuTensor& attention_mask,
                          GpuTensor& final_hidden_states,
                          BertEncoderCache& cache,
                          bool is_training) {
    if (!initial_hidden_states.is_allocated()) {
        throw std::runtime_error("Initial hidden_states not allocated for BertEncoder forward.");
    }
    
    // 캐시에 있는 두 개의 버퍼를 할당합니다.
    cache.hidden_states_buffer1.allocate(initial_hidden_states.dims_);
    cache.hidden_states_buffer2.allocate(initial_hidden_states.dims_);

    // input_ptr이 첫 번째 버퍼를, output_ptr이 두 번째 버퍼를 가리키게 합니다.
    GpuTensor* input_ptr = &cache.hidden_states_buffer1;
    GpuTensor* output_ptr = &cache.hidden_states_buffer2;
    
    // 모델의 첫 입력값을 버퍼 1로 복사합니다.
    input_ptr->copy_from_gpu(initial_hidden_states, stream);

    // 12개 레이어를 순회합니다.
    for (size_t i = 0; i < layers_.size(); ++i) {
        // BertLayer는 이제 외부에서 할당된 output_ptr 버퍼에 결과를 씁니다.
        layers_[i]->forward(blas_handle, stream, *input_ptr, attention_mask, *output_ptr, cache.layer_caches[i], is_training);
        
        // 다음 레이어의 입력을 위해 입력과 출력 포인터를 교체합니다 (핑퐁).
        std::swap(input_ptr, output_ptr);
    }
    
    // 마지막 레이어의 최종 출력(input_ptr에 저장됨)을 final_hidden_states로 복사합니다.
    final_hidden_states.copy_from_gpu(*input_ptr, stream);
}

// bert_components_hip.cpp 파일의 BertEncoder::backward를 교체하세요.

void BertEncoder::backward(rocblas_handle blas_handle, hipStream_t stream,
                           const GpuTensor& grad_final_hidden_states,
                           BertEncoderCache& cache,
                           GpuTensor& grad_initial_hidden_states) {
    std::cout << "[BertEncoder::backward] 시작 (Grad Output Ptr: " 
              << grad_final_hidden_states.d_ptr_ << ")" << std::endl;

    if (layers_.empty()) {
        grad_initial_hidden_states.copy_from_gpu(grad_final_hidden_states, stream);
        return;
    }
    
    grad_initial_hidden_states.allocate(grad_final_hidden_states.dims_);

    // [MODIFIED] 로컬 GpuTensor 대신 캐시에 미리 할당된 버퍼에 대한 포인터를 사용합니다.
    GpuTensor* current_grad_ptr = &cache.hidden_states_buffer1;
    GpuTensor* prev_layer_grad_ptr = &cache.hidden_states_buffer2;

    // 첫 그래디언트 입력을 버퍼 1로 복사합니다.
    current_grad_ptr->copy_from_gpu(grad_final_hidden_states, stream);

    // 마지막 레이어부터 첫 레이어까지 역순으로 순회합니다.
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        // current_grad_ptr를 입력으로, prev_layer_grad_ptr를 출력으로 하여 역전파를 수행합니다.
        layers_[i]->backward(blas_handle, stream, *current_grad_ptr, cache.layer_caches[i], *prev_layer_grad_ptr);
        
        // [MODIFIED] 비싼 copy_from_gpu 대신, 포인터만 교체하여 핑퐁(ping-pong) 방식으로 버퍼를 전환합니다.
        std::swap(current_grad_ptr, prev_layer_grad_ptr);
    }
    
    // 모든 레이어를 거친 최종 그래디언트(현재 current_grad_ptr에 저장됨)를
    // 함수의 최종 출력 텐서로 복사합니다.
    grad_initial_hidden_states.copy_from_gpu(*current_grad_ptr, stream);

    std::cout << "[BertEncoder::backward] 종료" << std::endl;
}

BertModel::BertModel(const BertConfig& config, const std::string& name_prefix)
    : config_(config),
      embeddings_(config, name_prefix + ".embeddings"),
      encoder_(config, name_prefix + ".encoder")
      {}

std::vector<Parameter*> BertModel::get_parameters() {
    auto embedding_params = embeddings_.get_parameters();
    auto encoder_params = encoder_.get_parameters();
    std::vector<Parameter*> params;
    params.insert(params.end(), embedding_params.begin(), embedding_params.end());
    params.insert(params.end(), encoder_params.begin(), encoder_params.end());
    return params;
}

Parameter* BertModel::get_word_embedding_params() {
    return embeddings_.get_word_embedding_params();
}

void BertModel::forward(rocblas_handle blas_handle, hipStream_t stream,
                        const GpuTensor& input_ids, const GpuTensor& attention_mask,
                        const GpuTensor& token_type_ids, GpuTensor& sequence_output,
                        BertModelCache& cache, bool is_training) {
    GpuTensor embedding_output;
    embedding_output.allocate({input_ids.dim_size(0), input_ids.dim_size(1), config_.hidden_size});
    embeddings_.forward(stream, input_ids, token_type_ids, embedding_output, cache.embeddings_cache, is_training);
    encoder_.forward(blas_handle, stream, embedding_output, attention_mask, sequence_output, cache.encoder_cache, is_training);
}

void BertModel::backward(rocblas_handle blas_handle, hipStream_t stream,
                         const GpuTensor& grad_sequence_output, BertModelCache& cache) {
    printf("[BertModel::backward] 시작 (Grad Output Ptr: %p)\n", grad_sequence_output.d_ptr_);

    // Encoder backward: grad_sequence_output -> grad_embedding_output
    GpuTensor grad_embedding_output;
    grad_embedding_output.allocate(grad_sequence_output.dims_);
    encoder_.backward(blas_handle, stream, grad_sequence_output, cache.encoder_cache, grad_embedding_output);

    // Embeddings backward: grad_embedding_output -> embeddings gradients
    embeddings_.backward(stream, grad_embedding_output, cache.embeddings_cache);
}
