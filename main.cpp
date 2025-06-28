#include <iostream>
#include <hip/hip_runtime.h>

int main() {
    try {
        std::cout << "HIP BERT Pre-training Starting..." << std::endl;
        
        // GPU 메모리 초기화 확인
        hipError_t hip_status = hipSetDevice(0);
        if (hip_status != hipSuccess) {
            std::cerr << "Failed to set HIP device: " << hipGetErrorString(hip_status) << std::endl;
            return -1;
        }
        
        std::cout << "HIP device initialized successfully." << std::endl;
        std::cout << "Program completed successfully." << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}