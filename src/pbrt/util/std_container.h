#pragma once

#include <vector>

template <typename T>
std::vector<T> sub_vector(const std::vector<T> &vec, int start, int end) {
    if (start >= vec.size()) {
        return {};
    }

    return std::vector(vec.begin() + start, vec.begin() + end);
}

template <typename T>
std::vector<T> sub_vector(const std::vector<T> &vec, int start) {
    if (start >= vec.size()) {
        return {};
    }

    return std::vector(vec.begin() + start, vec.end());
}
