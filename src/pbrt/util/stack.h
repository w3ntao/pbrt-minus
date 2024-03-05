#pragma once

#include "pbrt/util/macro.h"

template <typename T, int n>
class Stack {
  public:
    PBRT_GPU bool empty() const {
        return size == 0;
    }

    PBRT_GPU void push(const T &val) {
        if (size >= n) {
            printf("Stack::push() error: size (%d) >= limit (%d).", size, n);
            asm("trap;");
        }
        data[size] = val;
        size += 1;
    }

    PBRT_GPU T pop() {
        if (n <= 0) {
            printf("Stack::pop() error: no data in the stack.");
            asm("trap;");
        }
        size -= 1;
        return data[size];
    }

  private:
    T data[n];
    int size = 0;
};
