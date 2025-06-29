#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // std::vector, std::map 등 STL 컨테이너를 위한 헤더
#include <pybind11/numpy.h> // NumPy 배열과의 상호작용을 위한 헤더

#include "language_model_hip.hpp" // 메인 모델 클래스 헤더
#include "common_hip.hpp"       // GpuTensor 헤더
#include "bert_config.hpp"      // BertConfig 헤더

namespace py = pybind11;

// GpuTensor와 NumPy 배열 간의 변환을 위한 헬퍼 함수
// Python(NumPy) -> C++(GpuTensor)
void numpy_to_gputensor_int(GpuTensor& tensor, py::array_t<int, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim < 1) throw std::runtime_error("NumPy array must be at least 1-dimensional");

    std::vector<int> dims;
    for (int i = 0; i < buf.ndim; i++) {
        dims.push_back(buf.shape[i]);
    }
    tensor.allocate(dims);
    HIP_CHECK(hipMemcpy(tensor.d_ptr_, buf.ptr, tensor.size_in_bytes(), hipMemcpyHostToDevice));
}

void numpy_to_gputensor_float(GpuTensor& tensor, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim < 1) throw std::runtime_error("NumPy array must be at least 1-dimensional");

    std::vector<int> dims;
    for (int i = 0; i < buf.ndim; i++) {
        dims.push_back(buf.shape[i]);
    }
    tensor.allocate(dims);
    HIP_CHECK(hipMemcpy(tensor.d_ptr_, buf.ptr, tensor.size_in_bytes(), hipMemcpyHostToDevice));
}


// C++(GpuTensor) -> Python(NumPy)
py::array gputensor_to_numpy_int(const GpuTensor& tensor) {
    std::vector<ssize_t> shape;
    for (int dim : tensor.dims_) {
        shape.push_back(dim);
    }
    py::array_t<int> arr(shape);
    py::buffer_info buf = arr.request();
    HIP_CHECK(hipMemcpy(buf.ptr, tensor.d_ptr_, tensor.size_in_bytes(), hipMemcpyDeviceToHost));
    return arr;
}

py::array gputensor_to_numpy_float(const GpuTensor& tensor) {
    std::vector<ssize_t> shape;
    for (int dim : tensor.dims_) {
        shape.push_back(dim);
    }
    py::array_t<float> arr(shape);
    py::buffer_info buf = arr.request();
    HIP_CHECK(hipMemcpy(buf.ptr, tensor.d_ptr_, tensor.size_in_bytes(), hipMemcpyDeviceToHost));
    return arr;
}


// Python 모듈 정의
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
        .def_readwrite("hidden_dropout_prob", &BertConfig::hidden_dropout_prob)
        .def_readwrite("attention_probs_dropout_prob", &BertConfig::attention_probs_dropout_prob)
        .def_readwrite("layer_norm_eps", &BertConfig::layer_norm_eps);

    // GpuTensor 클래스는 Python에서 직접 생성하지 않고, 데이터 전달용으로만 사용합니다.
    // 따라서 NumPy와의 변환에 초점을 맞춥니다.
    py::class_<GpuTensor>(m, "GpuTensor");

    // CANBertForMaskedLM 클래스 바인딩
    py::class_<CANBertForMaskedLM>(m, "CANBertForMaskedLM")
        .def(py::init<const BertConfig&>(), py::arg("config")) // 생성자
        .def("initialize_parameters", &CANBertForMaskedLM::initialize_parameters,
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
        }, py::arg("input_ids"), py::arg("attention_mask"), py::arg("token_type_ids"), py::arg("labels"));
}