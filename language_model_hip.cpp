/***************************************
 *  language_model_hip.cpp  (final)    *
 ***************************************/
#include "language_model_hip.hpp"
#include "common_hip.hpp"
#include "hip_kernels.hpp"
#include "bert_components_hip.hpp"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <iomanip> // std::setprecision

/* ---------- 유틸 ---------- */
static void log_tensor(const std::string& name, const GpuTensor& t)
{
    if (!t.is_allocated()) {
        std::cout << "[LOG] " << name << ": <not-allocated>\n";
        return;
    }
    std::cout << "[LOG] " << name << ": dims=[";
    for (size_t i = 0; i < t.dims_.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << t.dims_[i];
    }
    std::cout << "] ptr=" << t.d_ptr_
              << " numel=" << t.num_elements_ << '\n';
}

/* ---------- BertLMPredictionHead ---------- */
BertLMPredictionHead::BertLMPredictionHead(
        const BertConfig& cfg,
        Parameter&        shared_word_emb_param,
        const std::string& name_prefix)
    : config_(cfg),
      transform_dense_(cfg, cfg.hidden_size, cfg.hidden_size,
                       name_prefix + ".transform.dense"),
      transform_layernorm_(cfg.hidden_size, cfg.layer_norm_eps,
                           name_prefix + ".transform.LayerNorm"),
      shared_word_embeddings_(shared_word_emb_param),
      decoder_bias_({cfg.vocab_size}, name_prefix + ".bias") {}

std::vector<Parameter*> BertLMPredictionHead::get_parameters()
{
    auto v = transform_dense_.get_parameters();
    auto ln = transform_layernorm_.get_parameters();
    v.insert(v.end(), ln.begin(), ln.end());
    v.push_back(&decoder_bias_);
    return v;
}

void BertLMPredictionHead::forward(
        rocblas_handle blas,
        hipStream_t    stream,
        const GpuTensor& hidden,
        GpuTensor&       logits,
        BertLMPredictionHeadCache& cache)
{
    const int B = hidden.dim_size(0);
    const int S = hidden.dim_size(1);
    const int M = B * S;
    const int K = config_.hidden_size;
    const int N = config_.vocab_size;

    /* 1) Dense → GELU → LayerNorm */
    cache.transform_dense_output.allocate(hidden.dims_);
    transform_dense_.forward(blas, stream, hidden,
                             cache.transform_dense_output,
                             cache.transform_dense_cache);

    cache.transform_gelu_output.allocate(hidden.dims_);
    transform_gelu_.forward(stream, cache.transform_dense_output,
                            cache.transform_gelu_output,
                            cache.transform_gelu_cache);

    cache.transform_layernorm_output.allocate(hidden.dims_);
   transform_layernorm_.forward(stream, cache.transform_gelu_output, // <-- 입력 텐서
                             cache.transform_layernorm_output,
                             cache.transform_layernorm_cache);

    /* 2) MatMul: (M×K) · (N×K)^T = (M×N) */
    logits.allocate({B, S, N});

    const float alpha = 1.f, beta = 0.f;
    ROCBLAS_CHECK( rocblas_set_stream(blas, stream) );
    ROCBLAS_CHECK( rocblas_sgemm(blas,
                                 rocblas_operation_none,
                                 rocblas_operation_transpose,
                                 M, N, K,
                                 &alpha,
                                 (const float*)cache.transform_layernorm_output.d_ptr_, M,   // lda
                                 (const float*)shared_word_embeddings_.weights.d_ptr_,  N,   // ldb (rows of B)
                                 &beta,
                                 (float*)logits.d_ptr_,                                   M) );// ldc

    /* 3) bias 추가 (GELU 없음) */
    launch_add_bias_only_kernel(stream,
                                (float*)logits.d_ptr_,
                                (const float*)logits.d_ptr_,
                                (const float*)decoder_bias_.weights.d_ptr_,
                                M, N);

    cache.hidden_states_input = &hidden;
}

void BertLMPredictionHead::backward(
        rocblas_handle blas,
        hipStream_t    stream,
        const GpuTensor& grad_logits,
        BertLMPredictionHeadCache& cache,
        GpuTensor& grad_hidden)
{
    const int B = cache.hidden_states_input->dim_size(0);
    const int S = cache.hidden_states_input->dim_size(1);
    const int M_vocab = config_.vocab_size;
    const int K_batch = B * S;
    const int N_hidden = config_.hidden_size;

    /* --- grad bias --- */
    launch_reduce_sum_kernel(stream,
                             (float*)decoder_bias_.grad_weights.d_ptr_,
                             (const float*)grad_logits.d_ptr_,
                             K_batch, M_vocab);

    /* --- grad shared embeddings (누적) --- */
    const float alpha = 1.f, beta_accum = 1.f;
    ROCBLAS_CHECK( rocblas_set_stream(blas, stream) );
    ROCBLAS_CHECK( rocblas_sgemm(blas,
                                 rocblas_operation_transpose,
                                 rocblas_operation_none,
                                 M_vocab, N_hidden, K_batch,
                                 &alpha,
                                 (const float*)grad_logits.d_ptr_,          K_batch,  // lda
                                 (const float*)cache.transform_layernorm_output.d_ptr_, K_batch, // ldb
                                 &beta_accum,
                                 (float*)shared_word_embeddings_.grad_weights.d_ptr_,
                                 M_vocab) );                                            // ldc

    /* --- grad w.r.t LayerNorm output --- */
/* --- grad w.r.t LayerNorm output ----------------------------------- */
cache.grad_transform_layernorm_output.allocate(
        cache.transform_layernorm_output.dims_);

const float beta_zero = 0.0f;      // 덮어쓰기

ROCBLAS_CHECK(
    rocblas_sgemm(
        blas,
        rocblas_operation_none,     // A : (K_batch × M_vocab)
        rocblas_operation_none,     // B : (M_vocab × N_hidden)
        K_batch,                    // m  = rows of C
        N_hidden,                   // n  = cols of C
        M_vocab,                    // k  = cols of A = rows of B
        &alpha,
        /* A */ (const float*)grad_logits.d_ptr_,              K_batch,   // lda
        /* B */ (const float*)shared_word_embeddings_.weights.d_ptr_,
                                                                  M_vocab, // ldb
        &beta_zero,
        /* C */ (float*)cache.grad_transform_layernorm_output.d_ptr_,
                                                                  K_batch  // ldc
    ));


    /* --- backward LayerNorm → GELU → Dense --- */
    cache.grad_transform_gelu_output.allocate(cache.transform_gelu_output.dims_);
    cache.grad_transform_dense_output.allocate(cache.transform_dense_output.dims_);
transform_layernorm_.backward(stream, cache.grad_transform_layernorm_output,
                              cache.transform_gelu_output, // <-- 이 텐서를 추가
                              cache.transform_layernorm_cache,
                              cache.grad_transform_gelu_output);

    transform_gelu_.backward(stream, cache.grad_transform_gelu_output,
                             cache.grad_transform_dense_output,
                             cache.transform_gelu_cache);

    transform_dense_.backward(blas, stream,
                              cache.grad_transform_dense_output,
                              cache.transform_dense_cache,
                              grad_hidden);
}

/* ---------- CANBertForMaskedLM ---------- */
CANBertForMaskedLM::CANBertForMaskedLM(const BertConfig& cfg)
    : config_(cfg)
{
    bert_model_ = std::make_unique<BertModel>(config_, "bert");
    lm_head_    = std::make_unique<BertLMPredictionHead>(
                    config_, *bert_model_->get_word_embedding_params(),
                    "cls.predictions");

    ROCBLAS_CHECK( rocblas_create_handle(&blas_handle_) );
    HIP_CHECK( hipStreamCreate(&stream_) );

    /* 파라미터 관리 & 옵티마이저 */
    all_params_ = get_parameters();
    for (auto* p : all_params_) if (p) p->allocate_gradients();

    optimizer_ = std::make_unique<AdamWOptimizer>(
                    all_params_, 1e-4f, 0.9f, 0.999f,
                    1e-8f, 1e-2f);

    /* 임시 버퍼 hipMalloc 1회 */

}

CANBertForMaskedLM::~CANBertForMaskedLM()
{
    if (stream_)       hipStreamDestroy(stream_);
    if (blas_handle_)  rocblas_destroy_handle(blas_handle_);
}

void CANBertForMaskedLM::initialize_parameters()
{
    for (auto* p : all_params_) {
        if (!p) continue;

        const std::string& name = p->name();   // 가정: Parameter에 name() getter

        if (name.find("LayerNorm.weight") != std::string::npos)
            p->initialize(Parameter::InitType::CONSTANT, 1.f);
        else if (name.find("LayerNorm.bias") != std::string::npos)
            p->initialize(Parameter::InitType::CONSTANT, 0.f);
        else if (name.rfind(".bias") != std::string::npos) // Dense bias 등
            p->initialize(Parameter::InitType::CONSTANT, 0.f);
        else
            p->initialize(Parameter::InitType::XAVIER_UNIFORM);
    }
}

std::vector<Parameter*> CANBertForMaskedLM::get_parameters()
{
    auto v = bert_model_->get_parameters();
    auto h = lm_head_->get_parameters();
    v.insert(v.end(), h.begin(), h.end());
    return v;
}
/* language_model_hip.cpp ------------------------------------------- *//* language_model_hip.cpp ------------------------------------------- */
void CANBertForMaskedLM::ensure_internal_buffers(int B, int S)
{
    const size_t need_seq = (size_t)B * S * config_.hidden_size;
    const size_t need_log = (size_t)B * S * config_.vocab_size;

    /* --------- sequence_output_ --------- */
    if (!sequence_output_.is_allocated() ||
        sequence_output_.num_elements_ < need_seq)
    {
        sequence_output_.allocate({B, S, config_.hidden_size});
    }

    /* grad_sequence_output_ MUST match sequence_output_ */
    if (!grad_sequence_output_.is_allocated() ||
        grad_sequence_output_.num_elements_ != sequence_output_.num_elements_)
    {
        grad_sequence_output_.allocate(sequence_output_.dims_);
    }

    /* --------- logits_ --------- */
    if (!logits_.is_allocated() ||
        logits_.num_elements_ < need_log)
    {
        logits_.allocate({B, S, config_.vocab_size});
    }

    /* grad_logits_ MUST match logits_  (★ 항상 검사/할당) */
    if (!grad_logits_.is_allocated() ||
        grad_logits_.num_elements_ != logits_.num_elements_)
    {
        grad_logits_.allocate(logits_.dims_);
    }
}



/*--------------------------------------------------------------------------*/
/*  작은 헬퍼: 시간 측정 & 로깅 매크로                                       */
/*--------------------------------------------------------------------------*/
#define DEBUG_TS(msg)                                                         \
    do {                                                                      \
        auto _now = std::chrono::high_resolution_clock::now();                \
        double _ms = std::chrono::duration<double, std::milli>(_now - t0).count();\
        std::cout << std::fixed << std::setprecision(2)                       \
                  << "[DEBUG " << std::setw(6) << _ms << " ms] " << msg       \
                  << std::endl;                                               \
    } while (0)

#define DEBUG_TENSOR(tag, T)                                                  \
    do {                                                                      \
        std::cout << "  " << tag << " : (";                                   \
        for (size_t _i = 0; _i < (T).dims_.size(); ++_i) {                    \
            std::cout << (T).dims_[_i] << (_i + 1 == (T).dims_.size() ? "" : "×");\
        }                                                                     \
        std::cout << ") ptr=" << (T).d_ptr_ << std::endl;                     \
    } while (0)

/*--------------------------------------------------------------------------*/
/*  CANBertForMaskedLM::train_step (with debugging)                          */
/*--------------------------------------------------------------------------*/
void CANBertForMaskedLM::train_step(const GpuTensor& input_ids,
                                    const GpuTensor& attention_mask,
                                    const GpuTensor& token_type_ids,
                                    const GpuTensor& labels,
                                    GpuTensor&       loss)
{
        // [추천] 이 로그를 추가하여 labels 텐서 상태를 바로 확인하세요.
    std::cout << "--- train_step 진입 직후 ---" << std::endl;
    log_tensor("DEBUG: input_ids", input_ids);
    log_tensor("DEBUG: labels", labels); // d_ptr_가 nullptr인지 확인
    /* 타이머 시작 */
    const auto t0 = std::chrono::high_resolution_clock::now();

    const int B = input_ids.dim_size(0);
    const int S = input_ids.dim_size(1);

    /* ❶ 내부 버퍼 확보 (없으면 hipMalloc) */
    ensure_internal_buffers(B, S);

    /*------------------------------------------------------------------------*/
    /* 0) 임시 버퍼 확보 / view                                                */
    /*------------------------------------------------------------------------*/
    GpuTensor seq_out, logits, grad_logits, grad_seq_out;

    /* 포인터는 미리 hipMalloc 해 둔 멤버를 사용하고,
       dim 정보만 실제 B×S 로 맞춰 줍니다. */
    seq_out.set_view(sequence_output_.d_ptr_,
                     {B, S, config_.hidden_size});

    logits.set_view(logits_.d_ptr_,
                    {B, S, config_.vocab_size});

    grad_logits.set_view(grad_logits_.d_ptr_,
                         logits.dims_);

    grad_seq_out.set_view(grad_sequence_output_.d_ptr_,
                          seq_out.dims_);

    if (!loss.is_allocated()) loss.allocate({1});
    loss.zero_out(stream_);

    DEBUG_TS("buffer/view prepared");

    /*------------------------------------------------------------------------*/
    /* 1) zero-grad                                                           */
    /*------------------------------------------------------------------------*/
    optimizer_->zero_grad(stream_);
    DEBUG_TS("optimizer.zero_grad");

    /*------------------------------------------------------------------------*/
    /* 2) Forward                                                             */
    /*------------------------------------------------------------------------*/
    BertModelCache            model_cache(config_.num_hidden_layers);
    BertLMPredictionHeadCache head_cache;

    DEBUG_TENSOR("input_ids", input_ids);
assert( grad_logits.is_allocated() && grad_logits.d_ptr_ );

    bert_model_->forward(blas_handle_, stream_,
                         input_ids, attention_mask, token_type_ids,
                         seq_out, model_cache, /*train=*/true);
    DEBUG_TS("bert_model forward");

    lm_head_->forward(blas_handle_, stream_, seq_out, logits, head_cache);
    DEBUG_TS("lm_head forward");

    /*------------------------------------------------------------------------*/
    /* 3) Loss + dLogits                                                      */
    /*------------------------------------------------------------------------*/
    loss.zero_out(stream_);
    launch_softmax_cross_entropy_loss_backward_optimized(
        stream_,
        (float*)grad_logits.d_ptr_,          // dL/dlogits
        (const float*)logits.d_ptr_,
        (const int*)labels.d_ptr_,
        (float*)loss.d_ptr_,
        B, S, config_.vocab_size, /*ignore_index=*/-100);

    HIP_CHECK( hipPeekAtLastError() );
    DEBUG_TS("loss + grad_logits");

    /*------------------------------------------------------------------------*/
    /* 4) Backward                                                            */
    /*------------------------/* --------- (추가) 텐서 차원·메모리 상태 로그 --------- */
std::cout << "input_ids dims: ";
for (int d : input_ids.dims_) std::cout << d << ' ';
std::cout << "\nnumel=" << input_ids.num_elements_
          << ", ptr=" << input_ids.d_ptr_ << std::endl;

size_t free_mem = 0, total_mem = 0;
HIP_CHECK( hipMemGetInfo(&free_mem, &total_mem) );
std::cout << "[GPU] free " << free_mem << " / total " << total_mem << " bytes"
          << std::endl;
/* --------- (추가) 텐서 차원·메모리 상태 로그 --------- */
  

    lm_head_->backward(blas_handle_, stream_,
                       grad_logits, head_cache, grad_seq_out);
    DEBUG_TS("lm_head backward");
assert( grad_logits.is_allocated() && grad_logits.d_ptr_ );

    bert_model_->backward(blas_handle_, stream_,
                          grad_seq_out, model_cache);
    DEBUG_TS("bert_model backward");

    /*------------------------------------------------------------------------*/
    /* 5) Optimizer step                                                      */
    /*------------------------------------------------------------------------*/
    
    optimizer_->step(stream_);
    DEBUG_TS("optimizer.step");

    /*------------------------------------------------------------------------*/
    /* 6) sync                                                                */
    /*------------------------------------------------------------------------*/
     std::cout << "[STEP] 최종 동기화 시작..." << std::endl;
    HIP_CHECK(hipStreamSynchronize(stream_));
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "[STEP] 최종 동기화 완료" << std::endl;
    
    std::cout << "===== train_step 정상 종료 =====\n" << std::endl;

    /* (선택) loss 값 호스트로 복사해 로그 */
    auto host_loss = loss.to_cpu<float>();
    if (!host_loss.empty())
        std::cout << "[LOSS] " << host_loss[0] << std::endl;
}
