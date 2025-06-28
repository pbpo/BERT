#include "workspace_manager.hpp"
#include <numeric>
#include <stdexcept>

WorkspaceManager::WorkspaceManager(size_t size_in_bytes, hipStream_t stream)
    : total_size_(size_in_bytes), stream_(stream) {
    if (total_size_ > 0) {
        workspace_buffer_ = std::make_unique<GpuTensor>("workspace_buffer");
        // Allocate as raw bytes
        workspace_buffer_->element_size_ = 1;
        workspace_buffer_->num_elements_ = total_size_;
        HIP_CHECK(hipMalloc(&workspace_buffer_->d_ptr_, total_size_));
        workspace_buffer_->allocated_ = true;
    }
}

GpuTensor WorkspaceManager::get_workspace(const std::vector<int>& dims, DataType type) {
    size_t num_elements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
    size_t element_size = (type == DataType::FLOAT32) ? sizeof(float) : sizeof(int);
    size_t required_size = num_elements * element_size;

    // Align to 256 bytes for performance
    const size_t alignment = 256;
    if (current_offset_ % alignment != 0) {
        current_offset_ += (alignment - (current_offset_ % alignment));
    }

    if (current_offset_ + required_size > total_size_) {
        throw std::runtime_error("Workspace memory exhausted. Required: " + std::to_string(required_size) +
                                 ", Available: " + std::to_string(total_size_ - current_offset_));
    }

    void* ptr = static_cast<char*>(workspace_buffer_->d_ptr_) + current_offset_;
    current_offset_ += required_size;

    GpuTensor view("workspace_view");
    view.set_view(ptr, dims, type);
    return view; // Return by value (move semantics will be used)
}

void WorkspaceManager::reset() {
    current_offset_ = 0;
}