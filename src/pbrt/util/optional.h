#pragma once

#include <pbrt/gpu/macro.h>

namespace pbrt {

template <typename T>
class optional {
  public:
    using value_type = T;

    PBRT_CPU_GPU
    optional() : set(false) {}

    PBRT_CPU_GPU
    optional(const T &v) : set(true) {
        new (ptr()) T(v);
    }

    PBRT_CPU_GPU
    optional(T &&v) : set(true) {
        new (ptr()) T(std::move(v));
    }

    PBRT_CPU_GPU
    optional(const optional &v) : set(v.has_value()) {
        if (v.has_value())
            new (ptr()) T(v.value());
    }

    PBRT_CPU_GPU
    optional(optional &&v) : set(v.has_value()) {
        if (v.has_value()) {
            new (ptr()) T(std::move(v.value()));
            v.reset();
        }
    }

    PBRT_CPU_GPU
    optional &operator=(const T &v) {
        reset();
        new (ptr()) T(v);
        set = true;
        return *this;
    }

    PBRT_CPU_GPU
    optional &operator=(T &&v) {
        reset();
        new (ptr()) T(std::move(v));
        set = true;
        return *this;
    }

    PBRT_CPU_GPU
    optional &operator=(const optional &v) {
        reset();
        if (v.has_value()) {
            new (ptr()) T(v.value());
            set = true;
        }
        return *this;
    }

    PBRT_CPU_GPU
    optional &operator=(optional &&v) {
        reset();
        if (v.has_value()) {
            new (ptr()) T(std::move(v.value()));
            set = true;
            v.reset();
        }
        return *this;
    }

    PBRT_CPU_GPU
    ~optional() {
        reset();
    }

    PBRT_CPU_GPU
    explicit operator bool() const {
        return set;
    }

    PBRT_CPU_GPU
    T value_or(const T &alt) const {
        return set ? value() : alt;
    }

    PBRT_CPU_GPU
    T *operator->() {
        return &value();
    }

    PBRT_CPU_GPU
    const T *operator->() const {
        return &value();
    }

    PBRT_CPU_GPU
    T &operator*() {
        return value();
    }

    PBRT_CPU_GPU
    const T &operator*() const {
        return value();
    }

    PBRT_CPU_GPU
    T &value() {
        return *ptr();
    }

    PBRT_CPU_GPU
    const T &value() const {
        return *ptr();
    }

    PBRT_CPU_GPU
    void reset() {
        if (set) {
            value().~T();
            set = false;
        }
    }

    PBRT_CPU_GPU
    bool has_value() const {
        return set;
    }

  private:
#ifdef __NVCC__
    // Work-around NVCC bug
    PBRT_CPU_GPU
    T *ptr() {
        return reinterpret_cast<T *>(&optionalValue);
    }

    PBRT_CPU_GPU
    const T *ptr() const {
        return reinterpret_cast<const T *>(&optionalValue);
    }

#else
    PBRT_CPU_GPU
    T *ptr() {
        return std::launder(reinterpret_cast<T *>(&optionalValue));
    }
    PBRT_CPU_GPU
    const T *ptr() const {
        return std::launder(reinterpret_cast<const T *>(&optionalValue));
    }
#endif

    std::aligned_storage_t<sizeof(T), alignof(T)> optionalValue;
    bool set = false;
};

} // namespace pbrt
