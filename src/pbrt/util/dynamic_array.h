#pragma once

#include "pbrt/util/macro.h"

template <typename T>
class DynamicArray {
    int used_size = 0;
    int capacity = 0;

    T *ptr_data = nullptr;

  public:
    PBRT_GPU DynamicArray() : used_size(0), capacity(0) {}

    PBRT_GPU DynamicArray(const DynamicArray &array)
        : used_size(array.used_size), capacity(array.capacity) {
        ptr_data = new T[capacity];
        memcpy(ptr_data, array.ptr_data, used_size);
    }

    PBRT_GPU ~DynamicArray() {
        delete[] ptr_data;
    }

    PBRT_GPU T operator[](size_t idx) const {
        return ptr_data[idx];
    }

    PBRT_GPU void push(T val) {
        if (capacity == 0) {
            capacity = 4;
            ptr_data = new T[capacity];

        } else if (used_size >= capacity) {
            // extend data
            capacity = std::min(capacity * 2, capacity + 4096);
            auto new_data = new T[capacity];
            memcpy(new_data, ptr_data, sizeof(T) * used_size);

            delete[] ptr_data;
            ptr_data = new_data;
        }
        // used_size < capacity

        ptr_data[used_size] = val;
        used_size += 1;
    }

    PBRT_GPU const T *data() const {
        return ptr_data;
    }

    PBRT_GPU int size() const {
        return used_size;
    }
};
