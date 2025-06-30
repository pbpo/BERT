#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <cmath>
#include <numeric>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h> // rocSOLVER 헤더 추가

// --- GPU 오류 체크를 위한 매크로 ---
#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s (%d) in %s at line %d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        throw std::runtime_error("HIP Error"); \
    } \
} while(0)

#define ROCBLAS_CHECK(cmd) do { \
    rocblas_status err = cmd; \
    if (err != rocblas_status_success) { \
        fprintf(stderr, "rocBLAS Error: %s (%d) in %s at line %d\n", rocblas_status_to_string(err), err, __FILE__, __LINE__); \
        throw std::runtime_error("rocBLAS Error"); \
    } \
} while(0)


// --- 헬퍼 클래스: GpuTensor ---
struct GpuTensor {
    float* d_ptr_ = nullptr;
    std::vector<int> dims_;
    bool is_view_ = false;

    ~GpuTensor() { free(); }
    GpuTensor() = default;
    GpuTensor(const GpuTensor&) = delete;
    GpuTensor& operator=(const GpuTensor&) = delete;
    GpuTensor(GpuTensor&& other) noexcept { move_from(std::move(other)); }
    GpuTensor& operator=(GpuTensor&& other) noexcept {
        if (this != &other) { free(); move_from(std::move(other)); }
        return *this;
    }

    void allocate(const std::vector<int>& dims) {
        free();
        dims_ = dims;
        is_view_ = false;
        size_t size = num_elements();
        if (size > 0) HIP_CHECK(hipMalloc(&d_ptr_, size * sizeof(float)));
    }
    
    void set_view(float* ptr, const std::vector<int>& dims) {
        free();
        d_ptr_ = ptr;
        dims_ = dims;
        is_view_ = true;
    }

    void free() {
        if (d_ptr_ != nullptr && !is_view_) HIP_CHECK(hipFree(d_ptr_));
        d_ptr_ = nullptr;
    }
    
    int dim_size(int i) const { return dims_.at(i); }
    size_t num_elements() const {
        if (dims_.empty()) return 0;
        return std::accumulate(dims_.begin(), dims_.end(), size_t(1), std::multiplies<size_t>());
    }
    bool is_allocated() const { return d_ptr_ != nullptr; }
    float* d_ptr() const { return d_ptr_; }

    void zero_out(hipStream_t stream) {
        if(is_allocated()) HIP_CHECK(hipMemsetAsync(d_ptr_, 0, num_elements() * sizeof(float), stream));
    }
private:
    void move_from(GpuTensor&& other) {
        d_ptr_ = other.d_ptr_;
        dims_ = std::move(other.dims_);
        is_view_ = other.is_view_;
        other.d_ptr_ = nullptr;
        other.is_view_ = false;
    }
};

// --- 헬퍼 클래스: Parameter ---
struct Parameter {
    GpuTensor weights;
    GpuTensor grad_weights;
};

// --- 커스텀 커널 및 래퍼 함수 ---
// C = alpha * A + beta * B
__global__ void matrix_set_kernel(float* C, const float* A, const float* B, float alpha, float beta, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = alpha * A[idx] + beta * B[idx];
}
// C = C + alpha * A
__global__ void matrix_add_inplace_kernel(float* C, const float* A, float alpha, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] += alpha * A[idx];
}
// Dst = Src^T
__global__ void transpose_copy_block_kernel(float* Dst, const float* Src, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) Dst[col * n + row] = Src[row * n + col];
}

void launch_matrix_set(hipStream_t stream, GpuTensor& C, float alpha, const GpuTensor& A, float beta, const GpuTensor& B) {
    size_t n = C.num_elements();
    hipLaunchKernelGGL(matrix_set_kernel, dim3((n + 255) / 256), dim3(256), 0, stream, C.d_ptr(), A.d_ptr(), B.d_ptr(), alpha, beta, n);
}
void launch_matrix_add_inplace(hipStream_t stream, GpuTensor& C, float alpha, const GpuTensor& A) {
    size_t n = C.num_elements();
    hipLaunchKernelGGL(matrix_add_inplace_kernel, dim3((n + 255) / 256), dim3(256), 0, stream, C.d_ptr(), A.d_ptr(), alpha, n);
}
void launch_transpose_copy_block(hipStream_t stream, GpuTensor& Dst, const GpuTensor& Src) {
    int n = Src.dim_size(0);
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    hipLaunchKernelGGL(transpose_copy_block_kernel, blocks, threads, 0, stream, Dst.d_ptr(), Src.d_ptr(), n);
}

static GpuTensor make_block_view(const GpuTensor& X, int r_idx, int c_idx, int r_bl, int c_bl, int ld_full) {
    float* ptr = X.d_ptr() + (size_t(r_idx) * r_bl * ld_full) + (size_t(c_idx) * c_bl);
    GpuTensor view;
    view.set_view(ptr, {r_bl, c_bl});
    return view;
}

// --- RXTX 알고리즘 구현 ---
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
    
    std::vector<GpuTensor> m(26), s(8), y(2), w(11), z(8), L(26), R(26);
    for (auto &t : m) t.allocate({ n_sub, n_sub });
    for (auto &t : s) t.allocate({ n_sub, n_sub });
    for (auto &t : y) t.allocate({ n_sub, m_sub });
    for (auto &t : w) t.allocate({ n_sub, m_sub });
    for (auto &t : z) t.allocate({ n_sub, n_sub });
    for (auto &t : L) t.allocate({ n_sub, m_sub });
    for (auto &t : R) t.allocate({ n_sub, m_sub });

    launch_matrix_set(stream, y[0],  1.0f, Xb[12], -1.0f, Xb[13]);
    launch_matrix_set(stream, y[1],  1.0f, Xb[11], -1.0f, Xb[9]);
    launch_matrix_set(stream, w[0],  1.0f, Xb[1], 1.0f, Xb[3]); launch_matrix_add_inplace(stream, w[0], -1.0f, Xb[7]);
    launch_matrix_set(stream, w[1],  1.0f, Xb[0], -1.0f, Xb[4]); launch_matrix_add_inplace(stream, w[1], -1.0f, Xb[5]);
    launch_matrix_set(stream, w[2],  1.0f, Xb[5], 1.0f, Xb[6]);
    launch_matrix_set(stream, w[3],  1.0f, Xb[13], 1.0f, Xb[14]);
    launch_matrix_set(stream, w[4],  1.0f, y[1], 1.0f, Xb[15]);
    launch_matrix_set(stream, w[5],  1.0f, Xb[9], 1.0f, Xb[10]);
    launch_matrix_set(stream, w[6],  1.0f, Xb[8], 1.0f, y[0]);
    launch_matrix_set(stream, w[7],  1.0f, Xb[8], -1.0f, Xb[7]);
    launch_matrix_set(stream, w[8],  1.0f, Xb[6], -1.0f, Xb[10]);
    launch_matrix_set(stream, w[9],  1.0f, Xb[5], -1.0f, Xb[6]);
    launch_matrix_set(stream, w[10], 1.0f, Xb[1], -1.0f, Xb[2]);

    launch_matrix_set(stream, L[0], -1.0f, w[0],  1.0f, Xb[2]);   launch_matrix_set(stream, R[0],  1.0f, Xb[7],   1.0f, Xb[10]);
    launch_matrix_set(stream, L[1],  1.0f, w[1],  1.0f, Xb[6]);   launch_matrix_set(stream, R[1],  1.0f, Xb[14],  1.0f, Xb[4]);
    launch_matrix_set(stream, L[2], -1.0f, Xb[1], 1.0f, Xb[11]);  launch_matrix_set(stream, R[2],  1.0f, w[4],    0.0f, R[2]);
    launch_matrix_set(stream, L[3],  1.0f, Xb[8],-1.0f, Xb[5]);   launch_matrix_set(stream, R[3],  1.0f, w[6],    0.0f, R[3]);
    launch_matrix_set(stream, L[4],  1.0f, Xb[1], 1.0f, Xb[10]);  launch_matrix_set(stream, R[4],  1.0f, Xb[14], -1.0f, w[2]);
    launch_matrix_set(stream, L[5],  1.0f, Xb[5], 1.0f, Xb[10]);  launch_matrix_set(stream, R[5],  1.0f, w[2],   -1.0f, Xb[10]);
    HIP_CHECK(hipMemcpyAsync(L[6].d_ptr(), Xb[10].d_ptr(), L[6].num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[6],  1.0f, w[2],    0.0f, R[6]);
    HIP_CHECK(hipMemcpyAsync(L[7].d_ptr(), Xb[1].d_ptr(),  L[7].num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[7],  1.0f, w[2],   -1.0f, w[3]); launch_matrix_add_inplace(stream, R[7], 1.0f, w[4]);
    HIP_CHECK(hipMemcpyAsync(L[8].d_ptr(), Xb[5].d_ptr(),  L[8].num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[8],  1.0f, w[6],   -1.0f, w[5]); launch_matrix_add_inplace(stream, R[8], 1.0f, w[2]);
    launch_matrix_set(stream, L[9],  1.0f, w[0], -1.0f, Xb[2]); launch_matrix_add_inplace(stream, L[9], 1.0f, Xb[6]); launch_matrix_add_inplace(stream, L[9],1.0f, Xb[10]);
    HIP_CHECK(hipMemcpyAsync(R[9].d_ptr(), Xb[10].d_ptr(), R[9].num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream));
    launch_matrix_set(stream, L[10], 1.0f, Xb[4],-1.0f, w[9]);   HIP_CHECK(hipMemcpyAsync(R[10].d_ptr(), Xb[4].d_ptr(), R[10].num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream));
    launch_matrix_set(stream, L[11], 1.0f, w[10],1.0f, Xb[3]);   HIP_CHECK(hipMemcpyAsync(R[11].d_ptr(), Xb[7].d_ptr(), R[11].num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream));
    launch_matrix_set(stream, L[12],-1.0f, w[1], 1.0f, Xb[2]);   HIP_CHECK(hipMemcpyAsync(R[12].d_ptr(), Xb[14].d_ptr(),R[12].num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream));
    launch_matrix_set(stream, L[13],-1.0f, w[1], 0.0f, L[13]);  launch_matrix_set(stream, R[13], 1.0f, w[6],    1.0f, w[3]);
    launch_matrix_set(stream, L[14], 1.0f, w[0], 0.0f, L[14]);   launch_matrix_set(stream, R[14], 1.0f, w[5],    1.0f, w[4]);
    launch_matrix_set(stream, L[15], 1.0f, Xb[0],-1.0f, Xb[7]);   launch_matrix_set(stream, R[15], 1.0f, Xb[8],  -1.0f, Xb[15]);
    HIP_CHECK(hipMemcpyAsync(L[16].d_ptr(), Xb[11].d_ptr(),L[16].num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[16],-1.0f, y[1],    0.0f, R[16]);
    HIP_CHECK(hipMemcpyAsync(L[17].d_ptr(), Xb[8].d_ptr(), L[17].num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[17], 1.0f, y[0],    0.0f, R[17]);
    launch_matrix_set(stream, L[18],-1.0f, w[10],0.0f, L[18]);  launch_matrix_set(stream, R[18],-1.0f, Xb[14],  1.0f, Xb[6]);
    launch_matrix_set(stream, L[19], 1.0f, Xb[4], 1.0f, w[7]);   HIP_CHECK(hipMemcpyAsync(R[19].d_ptr(), Xb[8].d_ptr(), R[19].num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(L[20].d_ptr(), Xb[7].d_ptr(), L[20].num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[20], 1.0f, Xb[11],  1.0f, w[7]);
    launch_matrix_set(stream, L[21],-1.0f, w[9], 0.0f, L[21]);  launch_matrix_set(stream, R[21], 1.0f, Xb[4],   1.0f, w[8]);
    HIP_CHECK(hipMemcpyAsync(L[22].d_ptr(), Xb[0].d_ptr(), L[22].num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[22], 1.0f, Xb[12], -1.0f, Xb[4]);
    HIP_CHECK(hipMemcpyAsync(L[23].d_ptr(), Xb[0].d_ptr(), L[23].num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[23], 1.0f, Xb[0],   1.0f, Xb[12]);
    launch_matrix_set(stream, L[24], 1.0f, Xb[8], 1.0f, Xb[1]);   HIP_CHECK(hipMemcpyAsync(R[24].d_ptr(), Xb[13].d_ptr(),R[24].num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream));
    launch_matrix_set(stream, L[25], 1.0f, Xb[5], 1.0f, Xb[9]);   HIP_CHECK(hipMemcpyAsync(R[25].d_ptr(), Xb[9].d_ptr(), R[25].num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream));
    
    const float alpha = 1.0f, beta = 0.0f;
    rocblas_set_stream(handle, stream);

    for (int i = 0; i < 26; ++i) {
        ROCBLAS_CHECK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                      n_sub, n_sub, m_sub, &alpha, L[i].d_ptr(), m_sub, R[i].d_ptr(), m_sub, &beta, m[i].d_ptr(), n_sub));
    }
    
    const int s_indices[] = {0, 1, 2, 3, 12, 13, 14, 15};
    for (int i = 0; i < 8; ++i) {
        ROCBLAS_CHECK(rocblas_ssyrk(handle, rocblas_fill_upper, rocblas_operation_none,
                      n_sub, m_sub, &alpha, Xb[s_indices[i]].d_ptr(), m_full, &beta, s[i].d_ptr(), n_sub));
    }

    auto Cblock = [&](int r, int c) { return make_block_view(C, r, c, n_sub, n_sub, n_full); };
    launch_matrix_set(stream, z[0],  1.0f, m[6],   -1.0f, m[10]); launch_matrix_add_inplace(stream, z[0], -1.0f, m[11]);
    launch_matrix_set(stream, z[1],  1.0f, m[0],    1.0f, m[11]); launch_matrix_add_inplace(stream, z[1],  1.0f, m[20]);
    launch_matrix_set(stream, z[2],  1.0f, m[2],    1.0f, m[16]); launch_matrix_add_inplace(stream, z[2], -1.0f, m[23]);
    launch_matrix_set(stream, z[3],  1.0f, m[1],    1.0f, m[10]); launch_matrix_add_inplace(stream, z[3],  1.0f, m[22]);
    launch_matrix_set(stream, z[4],  1.0f, m[4],    1.0f, m[6]);  launch_matrix_add_inplace(stream, z[4],  1.0f, m[7]);
    launch_matrix_set(stream, z[5],  1.0f, m[3],   -1.0f, m[17]); launch_matrix_add_inplace(stream, z[5], -1.0f, m[19]);
    launch_matrix_set(stream, z[6],  1.0f, m[5],   -1.0f, m[6]);  launch_matrix_add_inplace(stream, z[6], -1.0f, m[8]);
    launch_matrix_set(stream, z[7],  1.0f, m[16],   1.0f, m[17]);
    
    { auto t = Cblock(0,0); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, s[0], 1.0f, s[1]); launch_matrix_add_inplace(stream, t, 1.0f, s[2]); launch_matrix_add_inplace(stream, t, 1.0f, s[3]); }
    { auto t = Cblock(0,1); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, m[1], -1.0f, m[4]); launch_matrix_add_inplace(stream, t, -1.0f, z[0]); launch_matrix_add_inplace(stream, t, 1.0f, m[12]); launch_matrix_add_inplace(stream, t, 1.0f, m[18]); }
    { auto t = Cblock(0,2); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, z[1], 1.0f, z[2]); launch_matrix_add_inplace(stream, t, 1.0f, m[14]); launch_matrix_add_inplace(stream, t, 1.0f, m[15]); }
    { auto t = Cblock(0,3); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, z[3], -1.0f, z[2]); launch_matrix_add_inplace(stream, t, -1.0f, z[4]); launch_matrix_add_inplace(stream, t, 1.0f, m[12]); }
    { auto t = Cblock(1,1); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, m[0], 1.0f, m[5]); launch_matrix_add_inplace(stream, t, -1.0f, z[0]); launch_matrix_add_inplace(stream, t, 1.0f, m[9]); launch_matrix_add_inplace(stream, t, 1.0f, m[21]); }
    { auto t = Cblock(1,2); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, z[1], -1.0f, z[5]); launch_matrix_add_inplace(stream, t, 1.0f, z[6]); launch_matrix_add_inplace(stream, t, 1.0f, m[9]); }
    { auto t = Cblock(1,3); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, z[3], 1.0f, z[5]); launch_matrix_add_inplace(stream, t, 1.0f, m[13]); launch_matrix_add_inplace(stream, t, 1.0f, m[15]); }
    { auto t = Cblock(2,2); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, m[3], -1.0f, z[6]); launch_matrix_add_inplace(stream, t, -1.0f, z[7]); launch_matrix_add_inplace(stream, t, 1.0f, m[25]); }
    { auto t = Cblock(2,3); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, m[2], 1.0f, z[4]); launch_matrix_add_inplace(stream, t, 1.0f, z[7]); launch_matrix_add_inplace(stream, t, 1.0f, m[24]); }
    { auto t = Cblock(3,3); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, s[4], 1.0f, s[5]); launch_matrix_add_inplace(stream, t, 1.0f, s[6]); launch_matrix_add_inplace(stream, t, 1.0f, s[7]); }
    
    for (int i = 1; i < 4; ++i) {
        for (int j = 0; j < i; ++j) {
            auto C_ij = Cblock(i, j);
            auto C_ji = Cblock(j, i);
            launch_transpose_copy_block(stream, C_ij, C_ji);
        }
    }
}

// --- ShampooOptimizer가 사용하는 추가 커널 ---
__global__ void set_identity_kernel(float* matrix, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        matrix[row * n + col] = (row == col) ? 1.0f : 0.0f;
    }
}
__global__ void add_diagonal_kernel(float* matrix, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        matrix[i * n + i] += value;
    }
}
__global__ void elementwise_power_kernel(float* d, float exponent, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d[idx] = powf(d[idx], exponent);
    }
}
__global__ void scale_columns_kernel(float* out, const float* V, const float* d, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        // out(row, col) = V(row, col) * d(col)
        out[row * n + col] = V[row * n + col] * d[col];
    }
}


// --- Shampoo Optimizer 클래스 ---
class ShampooOptimizer {
private:
    std::vector<Parameter*>& params_;
    float lr_, beta2_, epsilon_;
    int update_freq_, t_ = 0;
    std::vector<GpuTensor> preconditioners_L_, preconditioners_R_;
    std::vector<GpuTensor> stats_GGT_, stats_GTG_;

    void launch_set_identity(hipStream_t stream, GpuTensor& t) {
        int n = t.dim_size(0);
        dim3 threads(16, 16);
        dim3 blocks((n + 15) / 16, (n + 15) / 16);
        hipLaunchKernelGGL(set_identity_kernel, blocks, threads, 0, stream, t.d_ptr(), n);
    }

    /**
     * @brief rocSOLVER를 사용하여 행렬의 -1/4 거듭제곱근을 정교하게 계산합니다.
     */
    void compute_matrix_inverse_root(hipStream_t stream, rocblas_handle handle, GpuTensor& out, const GpuTensor& in) {
        int n = in.dim_size(0);
        if (n == 0) return;
        
        rocblas_set_stream(handle, stream);

        GpuTensor A_copy, d_eigenvalues, e_superdiag;
        A_copy.allocate({n, n});
        d_eigenvalues.allocate({n});

        HIP_CHECK(hipMemcpyAsync(A_copy.d_ptr(), in.d_ptr(), in.num_elements() * sizeof(float), hipMemcpyDeviceToDevice, stream));
        
        // 대각선에 Epsilon을 더해 수치적 안정성 확보
        dim3 threads_diag(256);
        dim3 blocks_diag((n + 255) / 256);
        hipLaunchKernelGGL(add_diagonal_kernel, blocks_diag, threads_diag, 0, stream, A_copy.d_ptr(), n, epsilon_);

    rocblas_int devInfo;
    rocblas_status status = rocsolver_ssyevd(handle, 
                                             rocblas_evect_original, 
                                             rocblas_fill_upper, 
                                             n, 
                                             A_copy.d_ptr(), 
                                             n, 
                                             d_eigenvalues.d_ptr(), 
                                             e_superdiag.d_ptr(), 
                                             &devInfo);

        if (devInfo != 0) {
            // Eigendecomposition 실패 시 단위 행렬로 대체하여 안정적으로 진행
            launch_set_identity(stream, out);
            return;
        }

        // 2. 고유값에 -1/4 거듭제곱 적용: D' = D^(-1/4)
        hipLaunchKernelGGL(elementwise_power_kernel, blocks_diag, threads_diag, 0, stream, d_eigenvalues.d_ptr(), -0.25f, n);

        // 3. 행렬 재구성: H = V * D' * V^T
        GpuTensor temp; temp.allocate({n, n});
        
        // temp = V * D' (V의 열들을 D'의 원소로 스케일링)
        dim3 threads_2d(16, 16);
        dim3 blocks_2d((n + 15) / 16, (n + 15) / 16);
        hipLaunchKernelGGL(scale_columns_kernel, blocks_2d, threads_2d, 0, stream, temp.d_ptr(), A_copy.d_ptr(), d_eigenvalues.d_ptr(), n);

        // out = temp * V^T
        const float alpha = 1.0f, beta = 0.0f;
        ROCBLAS_CHECK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose, n, n, n, &alpha, temp.d_ptr(), n, A_copy.d_ptr(), n, &beta, out.d_ptr(), n));
    }


public:
    ShampooOptimizer(std::vector<Parameter*>& params, float lr = 1e-3, int update_freq = 20, float beta2=0.999f, float epsilon=1e-8f)
        : params_(params), lr_(lr), update_freq_(update_freq), beta2_(beta2), epsilon_(epsilon) {
        
        preconditioners_L_.resize(params_.size());
        preconditioners_R_.resize(params_.size());
        stats_GGT_.resize(params_.size());
        stats_GTG_.resize(params_.size());

        hipStream_t stream = 0;
        for (size_t i = 0; i < params.size(); ++i) {
            Parameter* p = params_[i];
            if (!p->weights.is_allocated()) continue;
            int n = p->weights.dim_size(0);
            int m = p->weights.dim_size(1);

            preconditioners_L_[i].allocate({n, n});
            preconditioners_R_[i].allocate({m, m});
            stats_GGT_[i].allocate({n, n});
            stats_GTG_[i].allocate({m, m});

            launch_set_identity(stream, preconditioners_L_[i]);
            launch_set_identity(stream, preconditioners_R_[i]);
            stats_GGT_[i].zero_out(stream);
            stats_GTG_[i].zero_out(stream);
        }
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    void step(hipStream_t stream, rocblas_handle handle) {
        t_++;
        rocblas_set_stream(handle, stream);

        for (size_t i = 0; i < params_.size(); ++i) {
            Parameter* p = params_[i];
            if (!p->weights.is_allocated() || !p->grad_weights.is_allocated()) continue;

            GpuTensor& grad = p->grad_weights;
            int n = grad.dim_size(0);
            int m = grad.dim_size(1);
            const float alpha = 1.0f, beta = 0.0f;

            GpuTensor ggt; ggt.allocate({n, n});
            if (n > 0 && m > 0) {
                if (n % 4 == 0 && m % 4 == 0 && n >= 4 && m >= 4) {
                    launch_rxtx_multiplication(stream, handle, ggt, grad);
                } else {
                    ROCBLAS_CHECK(rocblas_ssyrk(handle, rocblas_fill_upper, rocblas_operation_none, n, m, &alpha, grad.d_ptr(), m, &beta, ggt.d_ptr(), n));
                }
            }

            GpuTensor gtg; gtg.allocate({m, m});
            if (n > 0 && m > 0) {
                ROCBLAS_CHECK(rocblas_ssyrk(handle, rocblas_fill_upper, rocblas_operation_transpose, m, n, &alpha, grad.d_ptr(), m, &beta, gtg.d_ptr(), m));
            }
            
            launch_matrix_set(stream, stats_GGT_[i], 1.0f - beta2_, ggt, beta2_, stats_GGT_[i]);
            launch_matrix_set(stream, stats_GTG_[i], 1.0f - beta2_, gtg, beta2_, stats_GTG_[i]);
            
            if (t_ % update_freq_ == 0) {
                compute_matrix_inverse_root(stream, handle, preconditioners_L_[i], stats_GGT_[i]);
                compute_matrix_inverse_root(stream, handle, preconditioners_R_[i], stats_GTG_[i]);
            }

            GpuTensor preconditioned_grad; preconditioned_grad.allocate(grad.dims_);
            GpuTensor temp_grad; temp_grad.allocate({n, m});
            
            ROCBLAS_CHECK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                        n, m, n, &alpha, preconditioners_L_[i].d_ptr(), n, grad.d_ptr(), m, &beta, temp_grad.d_ptr(), n));
            
            ROCBLAS_CHECK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                        n, m, m, &alpha, temp_grad.d_ptr(), n, preconditioners_R_[i].d_ptr(), m, &beta, preconditioned_grad.d_ptr(), n));

            launch_matrix_add_inplace(stream, p->weights, -lr_, preconditioned_grad);
        }
    }
};

// --- AdamW Optimizer 커널 및 클래스 ---
__global__ void adamw_kernel(float* weights, const float* grad, float* m, float* v,
                             float lr, float beta1, float beta2, float epsilon, float weight_decay,
                             float beta1_t, float beta2_t, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weights[idx] -= lr * weight_decay * weights[idx];
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad[idx] * grad[idx];
        float m_hat = m[idx] / (1.0f - beta1_t);
        float v_hat = v[idx] / (1.0f - beta2_t);
        weights[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

class AdamWOptimizer {
private:
    std::vector<Parameter*>& params_;
    float lr_, beta1_, beta2_, epsilon_, weight_decay_;
    int t_ = 0;
    std::vector<GpuTensor> m_vec_, v_vec_;

public:
    AdamWOptimizer(std::vector<Parameter*>& params, float lr = 1e-3, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weight_decay = 1e-2f)
        : params_(params), lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), weight_decay_(weight_decay) {

        m_vec_.resize(params.size());
        v_vec_.resize(params.size());
        hipStream_t stream = 0;
        for (size_t i = 0; i < params.size(); ++i) {
            Parameter* p = params_[i];
            if (p->weights.is_allocated()) {
                m_vec_[i].allocate(p->weights.dims_);
                v_vec_[i].allocate(p->weights.dims_);
                m_vec_[i].zero_out(stream);
                v_vec_[i].zero_out(stream);
            }
        }
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    void step(hipStream_t stream) {
        t_++;
        float beta1_t = powf(beta1_, t_);
        float beta2_t = powf(beta2_, t_);

        for (size_t i = 0; i < params_.size(); ++i) {
            Parameter* p = params_[i];
            if (!p->weights.is_allocated() || !p->grad_weights.is_allocated()) continue;
            
            size_t n_elements = p->weights.num_elements();
            hipLaunchKernelGGL(adamw_kernel, dim3((n_elements + 255) / 256), dim3(256), 0, stream,
                               p->weights.d_ptr(), p->grad_weights.d_ptr(), m_vec_[i].d_ptr(), v_vec_[i].d_ptr(),
                               lr_, beta1_, beta2_, epsilon_, weight_decay_,
                               beta1_t, beta2_t, n_elements);
        }
    }
};


// --- 메인 함수 (옵티마이저 비교 테스트용) ---
int main() {
    try {
        hipStream_t stream;
        rocblas_handle handle;
        HIP_CHECK(hipStreamCreate(&stream));
        ROCBLAS_CHECK(rocblas_create_handle(&handle));

        const int N = 32;
        const int M = 64;
        const int STEPS = 100000;
        const float ADAM_LR = 1e-3;
        const float SHAMPOO_LR = 1e-5;

        Parameter p_shampoo, p_adamw;
        p_shampoo.weights.allocate({N, M});
        p_shampoo.grad_weights.allocate({N, M});
        p_adamw.weights.allocate({N, M});
        p_adamw.grad_weights.allocate({N, M});

        p_shampoo.weights.zero_out(stream);
        p_adamw.weights.zero_out(stream);
        
        std::vector<Parameter*> shampoo_params = {&p_shampoo};
        std::vector<Parameter*> adamw_params = {&p_adamw};

        ShampooOptimizer shampoo_opt(shampoo_params, SHAMPOO_LR, 5);
        AdamWOptimizer adamw_opt(adamw_params, ADAM_LR);

        std::cout << "ShampooOptimizer created." << std::endl;
        std::cout << "AdamWOptimizer created." << std::endl;
        
        std::cout << "\n--- Optimizer Battle Start! (" << STEPS << " Steps) ---" << std::endl;

        for (int step = 1; step <= STEPS; ++step) {
            std::cout << "Running optimizers step " << step << "..." << std::endl;
            std::vector<float> h_grad(N * M);
            for(size_t i = 0; i < h_grad.size(); ++i) h_grad[i] = (float(rand()) / float(RAND_MAX) - 0.5f);
            
            HIP_CHECK(hipMemcpy(p_shampoo.grad_weights.d_ptr(), h_grad.data(), h_grad.size() * sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(p_adamw.grad_weights.d_ptr(), h_grad.data(), h_grad.size() * sizeof(float), hipMemcpyHostToDevice));

            shampoo_opt.step(stream, handle);
            adamw_opt.step(stream);

            HIP_CHECK(hipStreamSynchronize(stream));
            std::cout << "Step " << step << " finished." << std::endl;
        }

        std::cout << "\n--- Final Results ---" << std::endl;
        float shampoo_norm = 0.0f;
        float adamw_norm = 0.0f;

        rocblas_set_stream(handle, stream);
        ROCBLAS_CHECK(rocblas_snrm2(handle, p_shampoo.weights.num_elements(), p_shampoo.weights.d_ptr(), 1, &shampoo_norm));
        ROCBLAS_CHECK(rocblas_snrm2(handle, p_adamw.weights.num_elements(), p_adamw.weights.d_ptr(), 1, &adamw_norm));

        HIP_CHECK(hipStreamSynchronize(stream));

        std::cout << "L2 Norm of Final Shampoo Weights: " << shampoo_norm << std::endl;
        std::cout << "L2 Norm of Final AdamW Weights  : " << adamw_norm << std::endl;

        ROCBLAS_CHECK(rocblas_destroy_handle(handle));
        HIP_CHECK(hipStreamDestroy(stream));
        std::cout << "\nProgram finished successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}