#ifndef NN_LAYERS_HIP_HPP
#define NN_LAYERS_HIP_HPP

#include "common_hip.hpp"
#include "bert_config.hpp"
#include <vector>
#include <string>
#include <iomanip> 
// --- Cache Structs ---
struct DropoutCache {
    GpuTensor mask;
};

struct DenseCache {
    const GpuTensor* input = nullptr;
};

struct LayerNormCache {
    // [FIXED] backward에 필요한 mean과 rstd를 저장할 GpuTensor
    GpuTensor mean;
    GpuTensor rstd;
   std::vector<int> input_dims; 
};

struct GeluCache {
    const GpuTensor* input = nullptr;
};

// --- Classes ---
class Gelu {
public:
    void forward(hipStream_t stream, const GpuTensor& input, GpuTensor& output, GeluCache& cache);
    void backward(hipStream_t stream, const GpuTensor& grad_output, GpuTensor& grad_input, const GeluCache& cache);
};

class Dropout {
private:
    float dropout_prob_;
    float scale_;  // scale_ 멤버 변수 추가

public:
    Dropout(float dropout_prob);  // 인라인 구현 제거
    void forward(hipStream_t stream, GpuTensor& input_output, DropoutCache& cache, bool is_training);
    void backward(hipStream_t stream, GpuTensor& grad_input, const GpuTensor& grad_output, const DropoutCache& cache);
    void backward(hipStream_t stream, GpuTensor& grad_io, const DropoutCache& cache);
};

class DenseLayer {
private:
    Parameter weights;
    Parameter bias;
    bool has_bias_;

public:
    DenseLayer(const BertConfig& config, int input_dim, int output_dim, const std::string& name_prefix, bool has_bias = true);
    std::vector<Parameter*> get_parameters();
    void allocate_gradients();
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& input, GpuTensor& output,
                 DenseCache& cache, bool is_training = false);
    
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_output, const DenseCache& cache,
                  GpuTensor& grad_input);
};


class LayerNorm : public Parameter {
public:
    // [FIXED] 생성자 선언을 (int, float, string)으로 통일
    LayerNorm(int hidden_size, float eps, const std::string& name_prefix);

    // [FIXED] 간단한 함수는 헤더에 바로 구현 (inline)
    std::vector<Parameter*> get_parameters() {
        return {this};
    }

    // [NEW] 누락되었던 함수 선언 추가
    void forward(hipStream_t stream, const GpuTensor& input, GpuTensor& output, LayerNormCache& cache);
   // original_input 파라미터를 추가하여 .cpp 파일의 정의와 동일하게 맞춰줍니다.
void backward(hipStream_t stream, const GpuTensor& grad_output, const GpuTensor& original_input, const LayerNormCache& cache, GpuTensor& grad_input);
    void allocate_gradients();

private:
    float eps_;
};
#endif // NN_LAYERS_HIP_HPP