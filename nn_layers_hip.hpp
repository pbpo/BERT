#ifndef NN_LAYERS_HIP_HPP
#define NN_LAYERS_HIP_HPP

#include "common_hip.hpp"
#include "bert_config.hpp"
#include <vector>
#include <string>

// --- Cache Structs ---
struct DropoutCache {
    GpuTensor mask;
};

struct DenseCache {
    const GpuTensor* input = nullptr;
};

struct LayerNormCache {
    const GpuTensor* input = nullptr;
    GpuTensor normalized_input;
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

class LayerNorm {
private:
    Parameter gamma;
    Parameter beta;
    float eps_;

public:
    LayerNorm(const BertConfig& config, const std::string& name_prefix);
    LayerNorm(int hidden_size, float eps = 1e-12f, const std::string& name_prefix = "LayerNorm");
    std::vector<Parameter*> get_parameters();
    void allocate_gradients();
    void forward(hipStream_t stream, const GpuTensor& input, GpuTensor& output, LayerNormCache& cache);
    void backward(hipStream_t stream, const GpuTensor& grad_output, const LayerNormCache& cache, GpuTensor& grad_input);
};

#endif // NN_LAYERS_HIP_HPP