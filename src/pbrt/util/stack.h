#pragma once

#include "pbrt/util/macro.h"

template <typename T, int Capacity>
class Stack {
  public:
    PBRT_GPU Stack() : size(0) {}

    PBRT_GPU bool empty() const {
        return size == 0;
    }

    PBRT_GPU void push(const T &val) {
        if (size >= Capacity) {
            printf("\nERROR: Stack::push(): size (%d) >= limit (%d).\n", size, Capacity);
            REPORT_FATAL_ERROR();
        }

        data[size] = val;
        size += 1;
    }

    PBRT_GPU T pop() {
        if (size == 0) {
            printf("\nERROR: Stack::pop(): no data in the stack.\n");
            REPORT_FATAL_ERROR();
        }

        size -= 1;
        return data[size];
    }

  private:
    T data[Capacity];
    uint size;
};
