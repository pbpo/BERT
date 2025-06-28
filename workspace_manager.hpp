#ifndef WORKSPACE_MANAGER_HPP
#define WORKSPACE_MANAGER_HPP

#include "common_hip.hpp"
#include <vector>
#include <memory>

class WorkspaceManager {
private:
    std::unique_ptr<GpuTensor> workspace_buffer_;
    size_t current_offset_ = 0;
    size_t total_size_ = 0;
    hipStream_t stream_;

public:
    // Pre-allocates a large buffer. Size is in bytes.
    WorkspaceManager(size_t size_in_bytes, hipStream_t stream);

    // Get a temporary tensor view from the workspace.
    GpuTensor get_workspace(const std::vector<int>& dims, DataType type = DataType::FLOAT32);

    // Reset the offset to reuse the workspace for the next iteration.
    void reset();

    // Get total allocated workspace size
    size_t get_total_size() const { return total_size_; }
};

#endif // WORKSPACE_MANAGER_HPP