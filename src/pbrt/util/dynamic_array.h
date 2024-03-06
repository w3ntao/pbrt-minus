#pragma once

#include "pbrt/util/macro.h"

template <typename T>
class DynamicArray {
    size_t used_size = 0;
    size_t capacity = 0;

    T *ptr_data = nullptr;

  public:
    PBRT_GPU DynamicArray() : used_size(0), capacity(0), ptr_data(nullptr) {}

    PBRT_GPU explicit DynamicArray(size_t _capacity) {
        used_size = 0;
        capacity = _capacity;
        ptr_data = new T[_capacity];
    }

    PBRT_GPU ~DynamicArray() {
        used_size = 0;
        capacity = 0;
        delete[] ptr_data;
        ptr_data = nullptr;
    }

    PBRT_GPU const T operator[](size_t idx) const {
        return ptr_data[idx];
    }

    PBRT_GPU void reserve(size_t _capacity) {
        if (_capacity <= capacity) {
            return;
        }

        auto new_data = new T[_capacity];
        memcpy(new_data, ptr_data, sizeof(T) * used_size);
        delete[] ptr_data;

        capacity = _capacity;
        ptr_data = new_data;
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
