#pragma once

#include "pbrt/util/macro.h"

template <typename T, int Capacity>
class Stack {
  public:
    PBRT_GPU bool empty() const {
        return size == 0;
    }

    PBRT_GPU void push(const T &val) {
        if (size >= Capacity) {
            printf("\n\n\nERROR: Stack::push(): size (%d) >= limit (%d).\n\n\n", size, Capacity);
            asm("trap;");
        }

        data[size] = val;
        size += 1;
    }

    PBRT_GPU T pop() {
        if (size <= 0) {
            printf("\n\n\nERROR: Stack::pop(): no data in the stack.\n\n\n");
            asm("trap;");
        }

        size -= 1;
        return data[size];
    }

  private:
    T data[Capacity];
    int size = 0;
};
