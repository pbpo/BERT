// bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// 프로젝트의 C++ 헤더 파일들
// 이 파일들이 모두 프로젝트 루트에 있다고 가정합니다.
#include "language_model_hip.hpp"
#include "common_hip.hpp"
#include "bert_config.hpp"

namespace py = pybind11;

// --- NumPy와 GpuTensor 간의 데이터 변환 헬퍼 함수 ---
// 이 함수들은 C++ 내부에서만 사용되므로 파이썬에 직접 노출할 필요는 없습니다.

// Python(NumPy) -> C++(GpuTensor)
void numpy_to_gputensor_int(GpuTensor& tensor, py::array_t<int, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim == 0) throw std::runtime_error("Input array cannot be a scalar.");
    std::vector<int> dims;
    for (int i = 0; i < buf.ndim; i++) {
        dims.push_back(buf.shape[i]);
    }
    tensor.allocate(dims); // GPU 메모리 할당
    HIP_CHECK(hipMemcpy(tensor.d_ptr_, buf.ptr, tensor.size_in_bytes(), hipMemcpyHostToDevice));
}

void numpy_to_gputensor_float(GpuTensor& tensor, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim == 0) throw std::runtime_error("Input array cannot be a scalar.");
    std::vector<int> dims;
    for (int i = 0; i < buf.ndim; i++) {
        dims.push_back(buf.shape[i]);
    }
    tensor.allocate(dims);
    HIP_CHECK(hipMemcpy(tensor.d_ptr_, buf.ptr, tensor.size_in_bytes(), hipMemcpyHostToDevice));
}

// C++(GpuTensor) -> Python(NumPy)
py::array gputensor_to_numpy_float(const GpuTensor& tensor) {
    std::vector<ssize_t> shape;
    // GpuTensor의 dims_가 비어있으면 스칼라로 처리 (예: 손실 값)
    if (tensor.dims_.empty() || (tensor.dims_.size() == 1 && tensor.dims_[0] == 1)) {
        shape.push_back(1);
    } else {
        for (int dim : tensor.dims_) {
            shape.push_back(dim);
        }
    }
    py::array_t<float> arr(shape);
    py::buffer_info buf = arr.request();
    HIP_CHECK(hipMemcpy(buf.ptr, tensor.d_ptr_, tensor.size_in_bytes(), hipMemcpyDeviceToHost));
    return arr;
}


// --- 파이썬 모듈 정의 ---
PYBIND11_MODULE(hipbert_core, m) {
    m.doc() = "HIP-based BERT implementation for Python"; // 모듈 설명

    // BertConfig 구조체 바인딩
    py::class_<BertConfig>(m, "BertConfig")
        .def(py::init<>()) // 기본 생성자
        .def_readwrite("vocab_size", &BertConfig::vocab_size)
        .def_readwrite("hidden_size", &BertConfig::hidden_size)
        .def_readwrite("num_hidden_layers", &BertConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &BertConfig::num_attention_heads)
        .def_readwrite("intermediate_size", &BertConfig::intermediate_size)
        .def_readwrite("max_position_embeddings", &BertConfig::max_position_embeddings)
        .def_readwrite("type_vocab_size", &BertConfig::type_vocab_size)
        .def_readwrite("hidden_act", &BertConfig::hidden_act)
        .def_readwrite("hidden_dropout_prob", &BertConfig::hidden_dropout_prob)
        .def_readwrite("attention_probs_dropout_prob", &BertConfig::attention_probs_dropout_prob)
        .def_readwrite("initializer_range", &BertConfig::initializer_range)
        .def_readwrite("layer_norm_eps", &BertConfig::layer_norm_eps);

    // CANBertForMaskedLM 클래스 바인딩
    py::class_<CANBertForMaskedLM>(m, "CANBertForMaskedLM")
        .def(py::init<const BertConfig&>(), "Constructor for the model, takes a BertConfig object.",
             py::arg("config"))
        .def("initialize_parameters", &CANBertForMaskedLM::initialize_parameters,
             "Initialize model parameters with a given mean and standard deviation.",
             py::arg("mean") = 0.0f, py::arg("stddev") = 0.02f)
        .def("train_step_numpy", [](CANBertForMaskedLM &self,
                                   py::array_t<int> input_ids,
                                   py::array_t<int> attention_mask,
                                   py::array_t<int> token_type_ids,
                                   py::array_t<int> labels) -> py::array {
            // 1. NumPy 배열을 GpuTensor로 변환
            GpuTensor input_ids_gpu("input_ids");
            GpuTensor attention_mask_gpu("attention_mask");
            GpuTensor token_type_ids_gpu("token_type_ids");
            GpuTensor labels_gpu("labels");
            GpuTensor loss_gpu("loss");

            numpy_to_gputensor_int(input_ids_gpu, input_ids);
            numpy_to_gputensor_int(attention_mask_gpu, attention_mask);
            numpy_to_gputensor_int(token_type_ids_gpu, token_type_ids);
            numpy_to_gputensor_int(labels_gpu, labels);
            
            // 2. C++의 train_step 함수 호출
            self.train_step(input_ids_gpu, attention_mask_gpu, token_type_ids_gpu, labels_gpu, loss_gpu);

            // 3. 결과 손실(loss)을 NumPy 배열로 변환하여 반환
            return gputensor_to_numpy_float(loss_gpu);
        }, "Performs one training step (forward, loss, backward, step) using NumPy arrays.",
           py::arg("input_ids"), py::arg("attention_mask"), py::arg("token_type_ids"), py::arg("labels"));
}