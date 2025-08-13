#pragma once

#include <mutex>
#include <pbrt/gpu/macro.h>
#include <vector>

class GPUMemoryAllocator {
  public:
    ~GPUMemoryAllocator() {
        std::unique_lock lock(mtx);

        for (auto ptr : gpu_dynamic_pointers) {
            CHECK_CUDA_ERROR(cudaFree(ptr));
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    template <typename T>
    [[nodiscard]] T *allocate(const size_t num = 1) {
        std::unique_lock lock(mtx);

        T *data;
        const auto size = sizeof(T) * num;
        CHECK_CUDA_ERROR(cudaMallocManaged(&data, size));
        gpu_dynamic_pointers.push_back(data);

        allocated_memory_size += size;

        return data;
    }

    [[nodiscard]] std::string get_allocated_memory_size() const;

  private:
    std::vector<void *> gpu_dynamic_pointers;
    ulong allocated_memory_size = 0;
    std::mutex mtx;
};
