#pragma once

#include <pbrt/gpu/macro.h>

class GPUMemoryAllocator;

class HashMap {
  public:
    static constexpr uint64_t EMPTY_KEY = 0xffffffff;

    struct KeyValue {
        uint64_t key;
        uint64_t value;
    };

    HashMap(int _capacity, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    uint64_t lookup(uint64_t key) const;

    PBRT_CPU_GPU
    void insert(uint64_t key, uint64_t value);

  private:
    KeyValue *items = nullptr;
    int size = 0;
    int capacity = 0;
};
