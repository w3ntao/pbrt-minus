#pragma once

#include <map>
#include <mutex>
#include <pbrt/gpu/macro.h>
#include <ranges>

class GPUMemoryAllocator {
  public:
    ~GPUMemoryAllocator() {
        std::unique_lock lock(mtx);

        for (const auto ptr : allocated_pointers | std::views::keys) {
            CHECK_CUDA_ERROR(cudaFree(const_cast<void *>(ptr)));
        }

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    template <typename T>
    [[nodiscard]]
    std::enable_if_t<!std::is_array_v<T>, T *> allocate(const size_t num = 1) {
        T *buffer;
        {
            const auto size = sizeof(T) * num;
            std::unique_lock lock(mtx);
            CHECK_CUDA_ERROR(cudaMallocManaged(&buffer, size));
            allocated_pointers[buffer] = size;
        }

        return buffer;
    }

    template <typename T, class... Params>
    [[nodiscard]]
    std::enable_if_t<!std::is_array_v<T>, T *> create(Params &&...params) {
        T *buffer;
        {
            const auto size = sizeof(T);
            std::unique_lock lock(mtx);
            CHECK_CUDA_ERROR(cudaMallocManaged(&buffer, size));
            allocated_pointers[buffer] = size;
        }

        return new (buffer) T(std::forward<Params>(params)...);
        // placement new:
        // https://en.cppreference.com/w/cpp/language/new.html
    }

    void release(const void *ptr) {
        std::unique_lock lock(mtx);

        const auto iterator = allocated_pointers.find(ptr);
        if (iterator == allocated_pointers.end()) {
            // pointer not found
            REPORT_FATAL_ERROR();
        }

        CHECK_CUDA_ERROR(cudaFree(const_cast<void *>(ptr)));

        allocated_pointers.erase(iterator);
    }

    [[nodiscard]]
    std::string get_allocated_memory_size() const;

  private:
    std::map<const void *, size_t> allocated_pointers;
    std::mutex mtx;
};
