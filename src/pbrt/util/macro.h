#pragma once

#include <iostream>

#define PBRT_CPU_GPU __host__ __device__
#define PBRT_GPU __device__

using FloatType = float;

namespace {
inline void _check_cuda(cudaError_t error_code, char const *const func, const char *const file,
                        int const line) {
    if (!error_code) {
        return;
    }

    std::cerr << "CUDA error at " << file << ": " << line << " '" << func << "'\n";
    auto error_str = cudaGetErrorString(error_code);
    std::cerr << "CUDA error " << static_cast<unsigned int>(error_code) << ": " << error_str
              << "\n";

    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(1);
}
} // namespace

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) _check_cuda((val), #val, __FILE__, __LINE__)

static const bool DEBUGGING = true;
