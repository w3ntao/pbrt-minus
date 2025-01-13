#pragma once

#include <pbrt/gpu/macro.h>

template <typename T, size_t Capacity>
class Stack {
  public:
    PBRT_CPU_GPU Stack() : size(0) {}

    PBRT_CPU_GPU size_t get_size() const {
        return size;
    }

    PBRT_CPU_GPU bool empty() const {
        return size <= 0;
    }

    PBRT_CPU_GPU void clear() {
        size = 0;
    }

    PBRT_CPU_GPU void push(const T &val) {
        if (size >= Capacity) {
            printf("\nERROR: Stack::push(): size (%ld) >= limit (%ld).\n", size, Capacity);
            REPORT_FATAL_ERROR();
        }

        data[size] = val;
        size += 1;
    }

    PBRT_CPU_GPU inline T pop() {
        if (size <= 0 || size > Capacity) {
            printf("\nERROR: Stack::pop(): no data in the stack.\n");
            REPORT_FATAL_ERROR();
        }

        size -= 1;
        return data[size];
    }

  private:
    T data[Capacity];
    size_t size;
};
