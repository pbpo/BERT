#include "optimizer_hip.hpp"
#include "common_hip.hpp"
#include "hip_kernels.hpp"
#include <rocsolver/rocsolver.h>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <unordered_map>

// rocSOLVER 오류 체크 매크로
#define ROCSOLVER_CHECK(cmd) do { \
    rocblas_status err = cmd; \
    if (err != rocblas_status_success) { \
        fprintf(stderr, "rocSOLVER Error: status %d in %s at line %d\n", err, __FILE__, __LINE__); \
        throw std::runtime_error("rocSOLVER Error"); \
    } \
} while(0)

// 누락된 함수들 구현
void launch_set_identity(hipStream_t stream, GpuTensor& tensor) {
    if (!tensor.is_allocated()) return;
    
    int n = tensor.dim_size(0);
    if (tensor.dims_.size() != 2 || tensor.dim_size(1) != n) return;
    
    // 단위 행렬 설정
    launch_set_identity_kernel(stream, (float*)tensor.d_ptr_, n);
}



void launch_add_diagonal_kernel(hipStream_t stream, float* matrix, int n, float value) {
    launch_add_diagonal_value_kernel(stream, matrix, n, value);
}

void launch_elementwise_power_kernel(hipStream_t stream, float* data, float power, size_t num_elements) {
    launch_power_kernel(stream, data, power, num_elements);
}

void launch_scale_columns_kernel(hipStream_t stream, float* output, const float* input, const float* scales, int n) {
    launch_matrix_scale_columns_kernel(stream, output, input, scales, n, n);
}


// RXTX 알고리즘에 필요한 헬퍼 함수
static GpuTensor make_block_view(const GpuTensor& X, int r_idx, int c_idx, int r_bl, int c_bl, int ld_full) {
    GpuTensor view;
    float* ptr = (float*)X.d_ptr_ + (size_t(r_idx) * r_bl * ld_full) + (size_t(c_idx) * c_bl);
    view.set_view(ptr, {r_bl, c_bl});
    return view;
}


// RXTX 알고리즘 구현 (d_ptr_float() -> d_ptr_ 수정)
void launch_rxtx_multiplication(hipStream_t stream, rocblas_handle handle, GpuTensor& C, const GpuTensor& X) {
    const int n_full = X.dim_size(0);
    const int m_full = X.dim_size(1);
    if (n_full % 4 != 0 || m_full % 4 != 0) {
        throw std::runtime_error("Matrix dimensions must be divisible by 4 for this RXTX implementation.");
    }
    const int n_sub  = n_full / 4;
    const int m_sub  = m_full / 4;

    std::vector<GpuTensor> Xb;
    Xb.reserve(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Xb.push_back(make_block_view(X, i, j, n_sub, m_sub, m_full));
        }
    }

    std::vector<GpuTensor> m, s, y, w, z, L, R;
    m.reserve(26); s.reserve(8); y.reserve(2); w.reserve(11); z.reserve(8); L.reserve(26); R.reserve(26);

    for (int i=0; i<26; ++i) m.emplace_back(std::vector<int>{n_sub, n_sub});
    for (int i=0; i<8; ++i) s.emplace_back(std::vector<int>{n_sub, n_sub});
    for (int i=0; i<2; ++i) y.emplace_back(std::vector<int>{n_sub, m_sub});
    for (int i=0; i<11; ++i) w.emplace_back(std::vector<int>{n_sub, m_sub});
    for (int i=0; i<8; ++i) z.emplace_back(std::vector<int>{n_sub, n_sub});
    for (int i=0; i<26; ++i) L.emplace_back(std::vector<int>{n_sub, m_sub});
    for (int i=0; i<26; ++i) R.emplace_back(std::vector<int>{n_sub, m_sub});

    // 모든 텐서 할당
    for (auto& tensor : m) tensor.allocate(tensor.dims_);
    for (auto& tensor : s) tensor.allocate(tensor.dims_);
    for (auto& tensor : y) tensor.allocate(tensor.dims_);
    for (auto& tensor : w) tensor.allocate(tensor.dims_);
    for (auto& tensor : z) tensor.allocate(tensor.dims_);
    for (auto& tensor : L) tensor.allocate(tensor.dims_);
    for (auto& tensor : R) tensor.allocate(tensor.dims_);

    // RXTX 알고리즘 연산들 - hip_kernels.cpp의 함수들 직접 사용
    launch_elementwise_add_kernel(stream, (float*)y[0].d_ptr_, (float*)Xb[12].d_ptr_, (float*)Xb[13].d_ptr_, y[0].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)y[1].d_ptr_, (float*)Xb[11].d_ptr_, (float*)Xb[9].d_ptr_, y[1].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)w[0].d_ptr_, (float*)Xb[1].d_ptr_, (float*)Xb[3].d_ptr_, w[0].num_elements_);
    launch_accumulate_kernel(stream, (float*)w[0].d_ptr_, (float*)Xb[7].d_ptr_, w[0].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)w[1].d_ptr_, (float*)Xb[0].d_ptr_, (float*)Xb[4].d_ptr_, w[1].num_elements_);
    launch_accumulate_kernel(stream, (float*)w[1].d_ptr_, (float*)Xb[5].d_ptr_, w[1].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)w[2].d_ptr_, (float*)Xb[5].d_ptr_, (float*)Xb[6].d_ptr_, w[2].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)w[3].d_ptr_, (float*)Xb[13].d_ptr_, (float*)Xb[14].d_ptr_, w[3].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)w[4].d_ptr_, (float*)y[1].d_ptr_, (float*)Xb[15].d_ptr_, w[4].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)w[5].d_ptr_, (float*)Xb[9].d_ptr_, (float*)Xb[10].d_ptr_, w[5].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)w[6].d_ptr_, (float*)Xb[8].d_ptr_, (float*)y[0].d_ptr_, w[6].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)w[7].d_ptr_, (float*)Xb[8].d_ptr_, (float*)Xb[7].d_ptr_, w[7].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)w[8].d_ptr_, (float*)Xb[6].d_ptr_, (float*)Xb[10].d_ptr_, w[8].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)w[9].d_ptr_, (float*)Xb[5].d_ptr_, (float*)Xb[6].d_ptr_, w[9].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)w[10].d_ptr_, (float*)Xb[1].d_ptr_, (float*)Xb[2].d_ptr_, w[10].num_elements_);

    // L과 R 행렬들 계산
    launch_elementwise_add_kernel(stream, (float*)L[0].d_ptr_, (float*)w[0].d_ptr_, (float*)Xb[2].d_ptr_, L[0].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)R[0].d_ptr_, (float*)Xb[7].d_ptr_, (float*)Xb[10].d_ptr_, R[0].num_elements_);
    
    launch_elementwise_add_kernel(stream, (float*)L[1].d_ptr_, (float*)w[1].d_ptr_, (float*)Xb[6].d_ptr_, L[1].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)R[1].d_ptr_, (float*)Xb[14].d_ptr_, (float*)Xb[4].d_ptr_, R[1].num_elements_);
    
    launch_elementwise_add_kernel(stream, (float*)L[2].d_ptr_, (float*)Xb[1].d_ptr_, (float*)Xb[11].d_ptr_, L[2].num_elements_);
    R[2].copy_from_gpu(w[4], stream);
    
    launch_elementwise_add_kernel(stream, (float*)L[3].d_ptr_, (float*)Xb[8].d_ptr_, (float*)Xb[5].d_ptr_, L[3].num_elements_);
    R[3].copy_from_gpu(w[6], stream);
    
    launch_elementwise_add_kernel(stream, (float*)L[4].d_ptr_, (float*)Xb[1].d_ptr_, (float*)Xb[10].d_ptr_, L[4].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)R[4].d_ptr_, (float*)Xb[14].d_ptr_, (float*)w[2].d_ptr_, R[4].num_elements_);
    
    launch_elementwise_add_kernel(stream, (float*)L[5].d_ptr_, (float*)Xb[5].d_ptr_, (float*)Xb[10].d_ptr_, L[5].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)R[5].d_ptr_, (float*)w[2].d_ptr_, (float*)Xb[10].d_ptr_, R[5].num_elements_);
    
    L[6].copy_from_gpu(Xb[10], stream);
    R[6].copy_from_gpu(w[2], stream);
    
    L[7].copy_from_gpu(Xb[1], stream);
    launch_elementwise_add_kernel(stream, (float*)R[7].d_ptr_, (float*)w[2].d_ptr_, (float*)w[3].d_ptr_, R[7].num_elements_);
    launch_accumulate_kernel(stream, (float*)R[7].d_ptr_, (float*)w[4].d_ptr_, R[7].num_elements_);
    
    L[8].copy_from_gpu(Xb[5], stream);
    launch_elementwise_add_kernel(stream, (float*)R[8].d_ptr_, (float*)w[6].d_ptr_, (float*)w[5].d_ptr_, R[8].num_elements_);
    launch_accumulate_kernel(stream, (float*)R[8].d_ptr_, (float*)w[2].d_ptr_, R[8].num_elements_);
    
    launch_elementwise_add_kernel(stream, (float*)L[9].d_ptr_, (float*)w[0].d_ptr_, (float*)Xb[2].d_ptr_, L[9].num_elements_);
    launch_accumulate_kernel(stream, (float*)L[9].d_ptr_, (float*)Xb[6].d_ptr_, L[9].num_elements_);
    launch_accumulate_kernel(stream, (float*)L[9].d_ptr_, (float*)Xb[10].d_ptr_, L[9].num_elements_);
    R[9].copy_from_gpu(Xb[10], stream);
    
    launch_elementwise_add_kernel(stream, (float*)L[10].d_ptr_, (float*)Xb[4].d_ptr_, (float*)w[9].d_ptr_, L[10].num_elements_);
    R[10].copy_from_gpu(Xb[4], stream);
    
    launch_elementwise_add_kernel(stream, (float*)L[11].d_ptr_, (float*)w[10].d_ptr_, (float*)Xb[3].d_ptr_, L[11].num_elements_);
    R[11].copy_from_gpu(Xb[7], stream);
    
    launch_elementwise_add_kernel(stream, (float*)L[12].d_ptr_, (float*)w[1].d_ptr_, (float*)Xb[2].d_ptr_, L[12].num_elements_);
    R[12].copy_from_gpu(Xb[14], stream);
    
    L[13].copy_from_gpu(w[1], stream);
    launch_elementwise_add_kernel(stream, (float*)R[13].d_ptr_, (float*)w[6].d_ptr_, (float*)w[3].d_ptr_, R[13].num_elements_);
    
    L[14].copy_from_gpu(w[0], stream);
    launch_elementwise_add_kernel(stream, (float*)R[14].d_ptr_, (float*)w[5].d_ptr_, (float*)w[4].d_ptr_, R[14].num_elements_);
    
    launch_elementwise_add_kernel(stream, (float*)L[15].d_ptr_, (float*)Xb[0].d_ptr_, (float*)Xb[7].d_ptr_, L[15].num_elements_);
    launch_elementwise_add_kernel(stream, (float*)R[15].d_ptr_, (float*)Xb[8].d_ptr_, (float*)Xb[15].d_ptr_, R[15].num_elements_);
    
    L[16].copy_from_gpu(Xb[11], stream);
    R[16].copy_from_gpu(y[1], stream);
    
    L[17].copy_from_gpu(Xb[8], stream);
    R[17].copy_from_gpu(y[0], stream);
    
    L[18].copy_from_gpu(w[10], stream);
    launch_elementwise_add_kernel(stream, (float*)R[18].d_ptr_, (float*)Xb[14].d_ptr_, (float*)Xb[6].d_ptr_, R[18].num_elements_);
    
    launch_elementwise_add_kernel(stream, (float*)L[19].d_ptr_, (float*)Xb[4].d_ptr_, (float*)w[7].d_ptr_, L[19].num_elements_);
    R[19].copy_from_gpu(Xb[8], stream);
    
    L[20].copy_from_gpu(Xb[7], stream);
    launch_elementwise_add_kernel(stream, (float*)R[20].d_ptr_, (float*)Xb[11].d_ptr_, (float*)w[7].d_ptr_, R[20].num_elements_);
    
    L[21].copy_from_gpu(w[9], stream);
    launch_elementwise_add_kernel(stream, (float*)R[21].d_ptr_, (float*)Xb[4].d_ptr_, (float*)w[8].d_ptr_, R[21].num_elements_);
    
    L[22].copy_from_gpu(Xb[0], stream);
    launch_elementwise_add_kernel(stream, (float*)R[22].d_ptr_, (float*)Xb[12].d_ptr_, (float*)Xb[4].d_ptr_, R[22].num_elements_);
    
    L[23].copy_from_gpu(Xb[0], stream);
    launch_elementwise_add_kernel(stream, (float*)R[23].d_ptr_, (float*)Xb[0].d_ptr_, (float*)Xb[12].d_ptr_, R[23].num_elements_);
    
    launch_elementwise_add_kernel(stream, (float*)L[24].d_ptr_, (float*)Xb[8].d_ptr_, (float*)Xb[1].d_ptr_, L[24].num_elements_);
    R[24].copy_from_gpu(Xb[13], stream);
    
    launch_elementwise_add_kernel(stream, (float*)L[25].d_ptr_, (float*)Xb[5].d_ptr_, (float*)Xb[9].d_ptr_, L[25].num_elements_);
    R[25].copy_from_gpu(Xb[9], stream);

    // 행렬 곱셈들
    const float alpha = 1.0f, beta = 0.0f;
    rocblas_set_stream(handle, stream);

    for (int i = 0; i < 26; ++i) {
        ROCBLAS_CHECK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                      n_sub, n_sub, m_sub, &alpha, 
                      (float*)L[i].d_ptr_, n_sub,
                      (float*)R[i].d_ptr_, n_sub,
                      &beta, (float*)m[i].d_ptr_, n_sub));
    }

    // SYRK 연산들
    const int s_indices[] = {0, 1, 2, 3, 12, 13, 14, 15};
    for (int i = 0; i < 8; ++i) {
        ROCBLAS_CHECK(rocblas_ssyrk(handle, rocblas_fill_upper, rocblas_operation_none,
                      n_sub, m_sub, &alpha, 
                      (float*)Xb[s_indices[i]].d_ptr_, n_sub,
                      &beta, (float*)s[i].d_ptr_, n_sub));
    }

    // z 벡터들 계산
    launch_elementwise_add_kernel(stream, (float*)z[0].d_ptr_, (float*)m[6].d_ptr_, (float*)m[10].d_ptr_, z[0].num_elements_);
    launch_accumulate_kernel(stream, (float*)z[0].d_ptr_, (float*)m[11].d_ptr_, z[0].num_elements_);
    
    launch_elementwise_add_kernel(stream, (float*)z[1].d_ptr_, (float*)m[0].d_ptr_, (float*)m[11].d_ptr_, z[1].num_elements_);
    launch_accumulate_kernel(stream, (float*)z[1].d_ptr_, (float*)m[20].d_ptr_, z[1].num_elements_);
    
    launch_elementwise_add_kernel(stream, (float*)z[2].d_ptr_, (float*)m[2].d_ptr_, (float*)m[16].d_ptr_, z[2].num_elements_);
    launch_accumulate_kernel(stream, (float*)z[2].d_ptr_, (float*)m[23].d_ptr_, z[2].num_elements_);
    
    launch_elementwise_add_kernel(stream, (float*)z[3].d_ptr_, (float*)m[1].d_ptr_, (float*)m[10].d_ptr_, z[3].num_elements_);
    launch_accumulate_kernel(stream, (float*)z[3].d_ptr_, (float*)m[22].d_ptr_, z[3].num_elements_);
    
    launch_elementwise_add_kernel(stream, (float*)z[4].d_ptr_, (float*)m[4].d_ptr_, (float*)m[6].d_ptr_, z[4].num_elements_);
    launch_accumulate_kernel(stream, (float*)z[4].d_ptr_, (float*)m[7].d_ptr_, z[4].num_elements_);
    
    launch_elementwise_add_kernel(stream, (float*)z[5].d_ptr_, (float*)m[3].d_ptr_, (float*)m[17].d_ptr_, z[5].num_elements_);
    launch_accumulate_kernel(stream, (float*)z[5].d_ptr_, (float*)m[19].d_ptr_, z[5].num_elements_);
    
    launch_elementwise_add_kernel(stream, (float*)z[6].d_ptr_, (float*)m[5].d_ptr_, (float*)m[6].d_ptr_, z[6].num_elements_);
    launch_accumulate_kernel(stream, (float*)z[6].d_ptr_, (float*)m[8].d_ptr_, z[6].num_elements_);
    
    launch_elementwise_add_kernel(stream, (float*)z[7].d_ptr_, (float*)m[16].d_ptr_, (float*)m[17].d_ptr_, z[7].num_elements_);

    auto Cblock = [&](int r, int c) { return make_block_view(C, r, c, n_sub, n_sub, n_full); };

    // 최종 결과 조립
    { auto t = Cblock(0,0); t.zero_out(stream); launch_elementwise_add_kernel(stream, (float*)t.d_ptr_, (float*)s[0].d_ptr_, (float*)s[1].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)s[2].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)s[3].d_ptr_, t.num_elements_); }
    { auto t = Cblock(0,1); t.zero_out(stream); launch_elementwise_add_kernel(stream, (float*)t.d_ptr_, (float*)m[1].d_ptr_, (float*)m[4].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)z[0].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)m[12].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)m[18].d_ptr_, t.num_elements_); }
    { auto t = Cblock(0,2); t.zero_out(stream); launch_elementwise_add_kernel(stream, (float*)t.d_ptr_, (float*)z[1].d_ptr_, (float*)z[2].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)m[14].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)m[15].d_ptr_, t.num_elements_); }
    { auto t = Cblock(0,3); t.zero_out(stream); launch_elementwise_add_kernel(stream, (float*)t.d_ptr_, (float*)z[3].d_ptr_, (float*)z[2].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)z[4].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)m[12].d_ptr_, t.num_elements_); }
    { auto t = Cblock(1,1); t.zero_out(stream); launch_elementwise_add_kernel(stream, (float*)t.d_ptr_, (float*)m[0].d_ptr_, (float*)m[5].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)z[0].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)m[9].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)m[21].d_ptr_, t.num_elements_); }
    { auto t = Cblock(1,2); t.zero_out(stream); launch_elementwise_add_kernel(stream, (float*)t.d_ptr_, (float*)z[1].d_ptr_, (float*)z[5].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)z[6].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)m[9].d_ptr_, t.num_elements_); }
    { auto t = Cblock(1,3); t.zero_out(stream); launch_elementwise_add_kernel(stream, (float*)t.d_ptr_, (float*)z[3].d_ptr_, (float*)z[5].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)m[13].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)m[15].d_ptr_, t.num_elements_); }
    { auto t = Cblock(2,2); t.zero_out(stream); launch_elementwise_add_kernel(stream, (float*)t.d_ptr_, (float*)m[3].d_ptr_, (float*)z[6].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)z[7].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)m[25].d_ptr_, t.num_elements_); }
    { auto t = Cblock(2,3); t.zero_out(stream); launch_elementwise_add_kernel(stream, (float*)t.d_ptr_, (float*)m[2].d_ptr_, (float*)z[4].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)z[7].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)m[24].d_ptr_, t.num_elements_); }
    { auto t = Cblock(3,3); t.zero_out(stream); launch_elementwise_add_kernel(stream, (float*)t.d_ptr_, (float*)s[4].d_ptr_, (float*)s[5].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)s[6].d_ptr_, t.num_elements_); launch_accumulate_kernel(stream, (float*)t.d_ptr_, (float*)s[7].d_ptr_, t.num_elements_); }

    // 대칭성을 위한 전치 복사
    for (int i = 1; i < 4; ++i) {
        for (int j = 0; j < i; ++j) {
            auto C_ij = Cblock(i, j);
            auto C_ji = Cblock(j, i);
            launch_transpose_back_kernel(stream, (float*)C_ij.d_ptr_, (float*)C_ji.d_ptr_, n_sub, n_sub, n_sub, n_sub);
        }
    }
}


// ShampooOptimizer 구현 (수정)
ShampooOptimizer::ShampooOptimizer(std::vector<Parameter*>& params, float lr, int update_freq, float beta2, float epsilon)
    : params_(params), lr_(lr), update_freq_(update_freq), beta2_(beta2), epsilon_(epsilon), t_(0) {

    hipStream_t stream = 0;
    for (Parameter* p : params) {
        if (!p || !p->weights.is_allocated()) continue;

        const GpuTensor* w_ptr = &p->weights;
        int n = w_ptr->dim_size(0);
        int m = w_ptr->dims_.size() > 1 ? w_ptr->dim_size(1) : 1;

        preconditioners_L_[w_ptr].allocate({n, n});
        preconditioners_R_[w_ptr].allocate({m, m});
        stats_GGT_[w_ptr].allocate({n, n});
        stats_GTG_[w_ptr].allocate({m, m});

        launch_set_identity(stream, preconditioners_L_.at(w_ptr));
        launch_set_identity(stream, preconditioners_R_.at(w_ptr));
        stats_GGT_.at(w_ptr).zero_out(stream);
        stats_GTG_.at(w_ptr).zero_out(stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
}


void ShampooOptimizer::compute_matrix_inverse_root(hipStream_t stream, rocblas_handle handle, GpuTensor& out, const GpuTensor& in) {
    int n = in.dim_size(0);
    if (n == 0) return;

    rocblas_set_stream(handle, stream);

    GpuTensor A_copy({n, n}, "A_copy");
    GpuTensor d_eigenvalues({n}, "eigenvalues");
    GpuTensor e_superdiag({n > 1 ? n - 1 : 1}, "superdiag");
    
    A_copy.allocate(A_copy.dims_);
    d_eigenvalues.allocate(d_eigenvalues.dims_);
    e_superdiag.allocate(e_superdiag.dims_);


    A_copy.copy_from_gpu(in, stream);
    launch_add_diagonal_kernel(stream, (float*)A_copy.d_ptr_, n, epsilon_);

    rocblas_int devInfo = 0;
    ROCSOLVER_CHECK(rocsolver_ssyevd(handle, rocblas_evect_original, rocblas_fill_upper, n, 
                                     (float*)A_copy.d_ptr_, n, (float*)d_eigenvalues.d_ptr_, 
                                     (float*)e_superdiag.d_ptr_, &devInfo));

    if (devInfo != 0) {
        launch_set_identity(stream, out);
        return;
    }

    launch_elementwise_power_kernel(stream, (float*)d_eigenvalues.d_ptr_, -0.25f, n);

    GpuTensor temp({n, n}, "temp");
        temp.allocate(temp.dims_);
    launch_scale_columns_kernel(stream, (float*)temp.d_ptr_, (float*)A_copy.d_ptr_, (float*)d_eigenvalues.d_ptr_, n);

    const float alpha = 1.0f, beta = 0.0f;
    ROCBLAS_CHECK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose, 
                                n, n, n, &alpha, (float*)temp.d_ptr_, n, 
                                (float*)A_copy.d_ptr_, n, &beta, (float*)out.d_ptr_, n));
}

void ShampooOptimizer::step(hipStream_t stream, rocblas_handle handle) {
    t_++;
    rocblas_set_stream(handle, stream);

    for (Parameter* p : params_) {
        if (!p || !p->weights.is_allocated() || !p->grad_weights.is_allocated()) continue;

        GpuTensor& grad = p->grad_weights;
        const GpuTensor* w_ptr = &p->weights;
        int n = grad.dim_size(0);
        int m = grad.dims_.size() > 1 ? grad.dim_size(1) : 1;

        const float alpha = 1.0f, beta = 0.0f;

        // GG^T 계산
        GpuTensor ggt({n, n}, "ggt");
        ggt.allocate(ggt.dims_);
        
        if (n > 0 && m > 0) {
            if (n % 4 == 0 && m % 4 == 0 && n >= 4 && m >= 4) {
                launch_rxtx_multiplication(stream, handle, ggt, grad);
            } else {
                ROCBLAS_CHECK(rocblas_ssyrk(handle, rocblas_fill_upper, rocblas_operation_none,
                                           n, m, &alpha, (float*)grad.d_ptr_, n,
                                           &beta, (float*)ggt.d_ptr_, n));
            }
        }

        // G^TG 계산
        GpuTensor gtg({m, m}, "gtg");
        gtg.allocate(gtg.dims_);  // 이 부분이 잘못되어 있었음 - ggt.allocate가 아니라 gtg.allocate
        
        if (n > 0 && m > 0) {
            ROCBLAS_CHECK(rocblas_ssyrk(handle, rocblas_fill_upper, rocblas_operation_transpose,
                                       m, n, &alpha, (float*)grad.d_ptr_, n,
                                       &beta, (float*)gtg.d_ptr_, m));
        }

        // 통계 업데이트
        launch_scale_kernel(stream, (float*)stats_GGT_.at(w_ptr).d_ptr_, beta2_, stats_GGT_.at(w_ptr).num_elements_);
        launch_accumulate_kernel(stream, (float*)stats_GGT_.at(w_ptr).d_ptr_, (float*)ggt.d_ptr_, stats_GGT_.at(w_ptr).num_elements_);
        
        launch_scale_kernel(stream, (float*)stats_GTG_.at(w_ptr).d_ptr_, beta2_, stats_GTG_.at(w_ptr).num_elements_);
        launch_accumulate_kernel(stream, (float*)stats_GTG_.at(w_ptr).d_ptr_, (float*)gtg.d_ptr_, stats_GTG_.at(w_ptr).num_elements_);

        if (t_ % update_freq_ == 0) {
            compute_matrix_inverse_root(stream, handle, preconditioners_L_.at(w_ptr), stats_GGT_.at(w_ptr));
            compute_matrix_inverse_root(stream, handle, preconditioners_R_.at(w_ptr), stats_GTG_.at(w_ptr));
        }

        GpuTensor preconditioned_grad(grad.dims_);
        GpuTensor temp_grad({n, m}, "temp_grad");
        preconditioned_grad.allocate(preconditioned_grad.dims_);
        temp_grad.allocate(temp_grad.dims_);

        const float gemm_alpha = 1.0f, gemm_beta = 0.0f;
        
        ROCBLAS_CHECK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, 
                                   n, m, n, &gemm_alpha, (float*)preconditioners_L_.at(w_ptr).d_ptr_, n, 
                                   (float*)grad.d_ptr_, n, &gemm_beta, (float*)temp_grad.d_ptr_, n));
        
        ROCBLAS_CHECK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, 
                                   n, m, m, &gemm_alpha, (float*)temp_grad.d_ptr_, n, 
                                   (float*)preconditioners_R_.at(w_ptr).d_ptr_, m, &gemm_beta, (float*)preconditioned_grad.d_ptr_, n));

        launch_scale_kernel(stream, (float*)preconditioned_grad.d_ptr_, -lr_, preconditioned_grad.num_elements_);
        launch_accumulate_kernel(stream, (float*)p->weights.d_ptr_, (float*)preconditioned_grad.d_ptr_, p->weights.num_elements_);
    }
}

// AdamWOptimizer 구현 (수정)
AdamWOptimizer::AdamWOptimizer(std::vector<Parameter*>& params, float lr, float beta1, float beta2, float epsilon, float weight_decay)
    : params_(params), lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), weight_decay_(weight_decay), t_(0) {

    hipStream_t stream = 0;
    for (Parameter* p : params) {
        if (!p) continue;
        if (p->weights.is_allocated()) {
            const GpuTensor* w_ptr = &p->weights;
            m_states_[w_ptr].allocate(w_ptr->dims_);
            v_states_[w_ptr].allocate(w_ptr->dims_);
            m_states_.at(w_ptr).zero_out(stream);
            v_states_.at(w_ptr).zero_out(stream);
        }
        if (p->has_bias_ && p->bias.is_allocated()) {
            const GpuTensor* b_ptr = &p->bias;
            m_states_[b_ptr].allocate(b_ptr->dims_);
            v_states_[b_ptr].allocate(b_ptr->dims_);
            m_states_.at(b_ptr).zero_out(stream);
            v_states_.at(b_ptr).zero_out(stream);
        }
    }
    HIP_CHECK(hipStreamSynchronize(stream));
}

void AdamWOptimizer::step(hipStream_t stream) {
    t_++;

    for (Parameter* p : params_) {
        if (!p) continue;

        if (p->weights.is_allocated() && p->grad_weights.is_allocated()) {
            const GpuTensor* w_ptr = &p->weights;
            launch_adamw_update_kernel(stream,
                (float*)p->weights.d_ptr_,
                (float*)p->grad_weights.d_ptr_,
                (float*)m_states_.at(w_ptr).d_ptr_,
                (float*)v_states_.at(w_ptr).d_ptr_,
                lr_, beta1_, beta2_, epsilon_, weight_decay_, t_, w_ptr->num_elements_);
        }

        if (p->has_bias_ && p->bias.is_allocated() && p->grad_bias.is_allocated()) {
             const GpuTensor* b_ptr = &p->bias;
             launch_adamw_update_kernel(stream,
                (float*)p->bias.d_ptr_,
                (float*)p->grad_bias.d_ptr_,
                (float*)m_states_.at(b_ptr).d_ptr_,
                (float*)v_states_.at(b_ptr).d_ptr_,
                lr_, beta1_, beta2_, epsilon_, 0.0f, t_, b_ptr->num_elements_);
        }
    }
}

void AdamWOptimizer::zero_grad(hipStream_t stream) {
    for (auto* p_param : params_) {
        if (p_param) {
            if (p_param->grad_weights.is_allocated()) {
                p_param->grad_weights.zero_out(stream);
            }
            if (p_param->has_bias_ && p_param->grad_bias.is_allocated()) {
                p_param->grad_bias.zero_out(stream);
            }
        }
    }
}