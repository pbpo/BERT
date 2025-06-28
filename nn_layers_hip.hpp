#pragma once

#include "common_hip.hpp"
#include "bert_config.hpp"
#include <vector>
#include <string>

// --- Forward Declarations ---
// No forward declarations needed here as this is the base

// --- Cache Structs ---
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

struct DropoutCache {
    GpuTensor mask;
    float dropout_prob;
};


// --- Layer Classes ---

class DenseLayer {
public:
    Parameter weights;
    Parameter bias;
    bool has_bias_;

    DenseLayer(const BertConfig& config, int input_dim, int output_dim, const std::string& name_prefix, bool has_bias = true);
    std::vector<Parameter*> get_parameters();
    void forward(rocblas_handle blas_handle, hipStream_t stream, const GpuTensor& input, GpuTensor& output, DenseCache& cache, bool is_training = false);
    void backward(rocblas_handle blas_handle, hipStream_t stream, const GpuTensor& grad_output, const DenseCache& cache, GpuTensor& grad_input);
};

class LayerNorm {
public:
    Parameter gamma;
    Parameter beta;
    float eps_;

    LayerNorm(const BertConfig& config, const std::string& name_prefix);
    LayerNorm(int hidden_size, float eps, const std::string& name_prefix); // Overloaded constructor
    std::vector<Parameter*> get_parameters();
    void forward(hipStream_t stream, const GpuTensor& input, GpuTensor& output, LayerNormCache& cache);
    void backward(hipStream_t stream, const GpuTensor& grad_output, const LayerNormCache& cache, GpuTensor& grad_input);
};

class Gelu {
public:
    void forward(hipStream_t stream, const GpuTensor& input, GpuTensor& output, GeluCache& cache);
};

class Dropout {
public:
    float dropout_prob_;
    Dropout(float dropout_prob) : dropout_prob_(dropout_prob) {}
    void forward(hipStream_t stream, GpuTensor& io, DropoutCache& cache, bool is_training);
    void backward(hipStream_t stream, GpuTensor& grad_io, const DropoutCache& cache);
};