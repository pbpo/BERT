// main.cpp
#include <iostream>
#include <random>
#include <vector>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "bert_config.hpp"
#include "language_model_hip.hpp"   // CANBertForMaskedLM 정의
#include "common_hip.hpp"           // GpuTensor, HIP_CHECK 등
#include <unistd.h>  
int main(int argc, char** argv) {
        FILE* log_fp = std::fopen("train.log", "w");
         
    if (!log_fp) {
        std::perror("fopen(train.log) failed");
        return EXIT_FAILURE;
    }
        bool only_init = (argc>1 && std::string(argv[1])=="--init-only");
    hipError_t err = hipSetDevice(0);
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to set HIP device: %s\n", hipGetErrorString(err));
        return -1;
    }
        std::fflush(stdout);
    std::fflush(stderr);
    dup2(fileno(log_fp), STDOUT_FILENO);
    dup2(fileno(log_fp), STDERR_FILENO);

    // 이후의 printf, std::cout, fprintf(stderr,…) 모두 train.log에 기록됩니다.
    std::printf("CAN-BERT HIP pre-training demo\n");
    try {
        std::cout << "CAN-BERT HIP pre-training demo\n";

        // ---------------------------------------------------------------------
        // 1. HIP 장치 초기화
        // ---------------------------------------------------------------------
        HIP_CHECK(hipSetDevice(0));
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, 0));
        std::cout << "Using GPU : " << prop.name << '\n';

        // ---------------------------------------------------------------------
        // 2. BertConfig 설정 (필요 시 값 수정)
        // ---------------------------------------------------------------------
        BertConfig cfg;
        cfg.vocab_size                    = 30522;
        cfg.hidden_size                   = 768;
        cfg.num_hidden_layers             = 12;
        cfg.num_attention_heads           = 12;
        cfg.intermediate_size             = 3072;
        cfg.max_position_embeddings       = 512;
        cfg.type_vocab_size               = 2;
        cfg.hidden_act                    = "gelu";
        cfg.hidden_dropout_prob           = 0.1f;
        cfg.attention_probs_dropout_prob  = 0.1f;
        cfg.initializer_range             = 0.02f;

        // Explicit cleanup scope to ensure proper destruction order
        {
            // ---------------------------------------------------------------------
            // 3. 모델 생성 및 파라미터 초기화
            // ---------------------------------------------------------------------
            CANBertForMaskedLM model(cfg);
            model.initialize_parameters();


            // ---------------------------------------------------------------------
            // 4. 더미 배치 준비 (B=8, S=128)
            // ---------------------------------------------------------------------
            constexpr int B = 8;
            constexpr int S = 128;

            std::mt19937 rng(42);
            std::uniform_int_distribution<int> token_dist(0, cfg.vocab_size - 1);

            std::vector<int> h_input_ids(B * S);
            std::vector<int> h_labels(B * S);
            std::vector<int> h_attention_mask(B * S, 1);
            std::vector<int> h_token_type_ids(B * S, 0);

            for (size_t i = 0; i < h_input_ids.size(); ++i) {
                int id          = token_dist(rng);
                h_input_ids[i]  = id;
                h_labels[i]     = id;
            }

            // GPU Tensor scope for controlled destruction
            {
                GpuTensor d_input_ids   ({B, S}, "input_ids",        DataType::INT32);
                GpuTensor d_attn_mask   ({B, S}, "attention_mask",   DataType::INT32);
                GpuTensor d_token_types ({B, S}, "token_type_ids",   DataType::INT32);
                GpuTensor d_labels      ({B, S}, "labels",           DataType::INT32);
                GpuTensor d_loss        ("loss");

                d_input_ids.to_gpu   (h_input_ids);
                d_attn_mask.to_gpu   (h_attention_mask);
                d_token_types.to_gpu (h_token_type_ids);
                d_labels.to_gpu      (h_labels);
                                // 텐서 생성 직후 차원 확인 로그 추가
                fprintf(stderr, "텐서 생성 후 차원 확인:\n");
                fprintf(stderr, "  d_input_ids: dims.size()=%zu", d_input_ids.dims_.size());
                fprintf(stderr, ", dims=[");
                for (size_t i = 0; i < d_input_ids.dims_.size(); ++i) {
                    if (i > 0) fprintf(stderr, ", ");
                    fprintf(stderr, "%d", d_input_ids.dims_[i]);
                }
                fprintf(stderr, "], num_elements=%zu\n", d_input_ids.num_elements_);
                
                // 상수 확인
                fprintf(stderr, "상수 확인: B=%d, S=%d, B*S=%d\n", B, S, B*S);

                // HIP 상태 확인
                fprintf(stderr, "데이터 업로드 후 HIP 상태 확인...\n");
                hipError_t err_check1 = hipGetLastError();
                if (err_check1 != hipSuccess) {
                    fprintf(stderr, "데이터 업로드 후 HIP 오류: %s\n", hipGetErrorString(err_check1));
                }
                
                fprintf(stderr, "hipDeviceSynchronize 호출...\n");
                HIP_CHECK(hipDeviceSynchronize());
                fprintf(stderr, "동기화 완료.\n");

                // ---------------------------------------------------------------------
                // 5. 단일 학습 스텝 실행
       const int kNumEpochs = 2;             // ★ 원하는 epoch 수
for (int epoch = 0; epoch < kNumEpochs; ++epoch)
{
    fprintf(stderr, "\n========== Epoch %d / %d ==========\n",
            epoch + 1, kNumEpochs);

    // (1) optional: 데이터·라벨 재셔플/마스킹 로직을 넣으려면 여기서
    //     h_input_ids / h_labels 등을 다시 채운 뒤 d_input_ids.to_gpu(…) 호출

    try {
        fprintf(stderr, "model.train_step() 호출 시작...\n");
        model.train_step(d_input_ids,
                         d_attn_mask,
                         d_token_types,
                         d_labels,
                         d_loss);
        fprintf(stderr, "model.train_step() 호출 완료!\n");
    } catch (const std::exception& e) {
        fprintf(stderr, "train_step에서 예외 발생: %s\n", e.what());
        throw;
    }

    HIP_CHECK(hipDeviceSynchronize());

    // (2) 손실 로그
    auto h_loss = d_loss.to_cpu<float>();
    if (!h_loss.empty())
        std::cout << "[Epoch " << (epoch + 1) << "] Loss : "
                  << h_loss[0] << '\n';
}
std::cout << "Training loop (“" << kNumEpochs << " epoch”) completed.\n";

                HIP_CHECK(hipDeviceSynchronize());

                // ---------------------------------------------------------------------
                // 6. 손실값 출력
                // ---------------------------------------------------------------------
                auto h_loss = d_loss.to_cpu<float>();
                if (!h_loss.empty()) std::cout << "Loss  : " << h_loss[0] << '\n';

                std::cout << "Training step completed successfully.\n";
                
                // Explicit cleanup in proper order
                fprintf(stderr, "Starting tensor cleanup...\n");
                
                // Clear any pending HIP errors
                hipGetLastError();
                
                // Synchronize before cleanup
                hipDeviceSynchronize();
                
                fprintf(stderr, "Tensors going out of scope...\n");
            } // GPU tensors destroyed here
            
            fprintf(stderr, "Model going out of scope...\n");
        } // Model destroyed here
        
        fprintf(stderr, "Local cleanup completed, calling hipDeviceReset...\n");
        
        // Final cleanup with error checking
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            fprintf(stderr, "Warning: hipDeviceSynchronize failed: %s\n", hipGetErrorString(err));
        }
        
        err = hipDeviceReset();
        if (err != hipSuccess) {
            fprintf(stderr, "Warning: hipDeviceReset failed: %s\n", hipGetErrorString(err));
        } else {
            fprintf(stderr, "hipDeviceReset completed successfully.\n");
        }
        
        fprintf(stderr, "Program terminating normally.\n");
            std::fclose(log_fp);

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Exception : " << ex.what() << '\n';
        
        // More detailed error handling in catch block
        fprintf(stderr, "Attempting cleanup after exception...\n");
        
        // Clear any pending errors before cleanup
        hipError_t lastErr = hipGetLastError();
        if (lastErr != hipSuccess) {
            fprintf(stderr, "Pending HIP error: %s\n", hipGetErrorString(lastErr));
        }
        
        hipError_t syncErr = hipDeviceSynchronize();
        if (syncErr != hipSuccess) {
            fprintf(stderr, "hipDeviceSynchronize in catch failed: %s\n", hipGetErrorString(syncErr));
        }
        
        hipError_t resetErr = hipDeviceReset();
        if (resetErr != hipSuccess) {
            fprintf(stderr, "hipDeviceReset in catch failed: %s\n", hipGetErrorString(resetErr));
        }
            std::fclose(log_fp);
 
        return -1;
    }
}