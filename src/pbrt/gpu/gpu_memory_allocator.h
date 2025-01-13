#pragma once

#include <pbrt/gpu/macro.h>
#include <vector>

class GPUMemoryAllocator {
  public:
    ~GPUMemoryAllocator() {
        for (auto ptr : gpu_dynamic_pointers) {
            CHECK_CUDA_ERROR(cudaFree(ptr));
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    template <typename T>
    T *allocate(const size_t num = 1) {
        T *data;
        CHECK_CUDA_ERROR(cudaMallocManaged(&data, sizeof(T) * num));
        gpu_dynamic_pointers.push_back(data);

        return data;
    }

  private:
    std::vector<void *> gpu_dynamic_pointers;
};
