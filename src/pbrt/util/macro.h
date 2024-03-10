#pragma once

#include <iostream>

#define PBRT_CPU_GPU __host__ __device__
#define PBRT_GPU __device__

namespace {
inline void _check_cuda(cudaError_t result, char const *const func, const char *const file,
                        int const line) {
    if (!result) {
        return;
    }

    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":"
              << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(-1);
}
} // namespace
// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) _check_cuda((val), #val, __FILE__, __LINE__)
