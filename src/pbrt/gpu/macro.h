#pragma once

#include <iostream>
#include <stdint.h>
// don't delete `stdint.h` as some compilers need it for `uint8_t`

#define PBRT_CPU_GPU __host__ __device__
#define PBRT_GPU __device__

#ifdef PBRT_FLOAT_AS_DOUBLE
using Real = double;
#else
using Real = float;
#endif

#define FLAG_COLORFUL_PRINT_RED_START "\033[0;31m"
#define FLAG_COLORFUL_PRINT_END "\033[0m"

namespace HIDDEN {
static void _check_cuda_error(const cudaError_t error_code, char const *const func,
                              const char *const file, int const line) {
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

PBRT_CPU_GPU
static void _report_error(const char *file_name, const char *func_name, const uint line_num) {
    printf("\nERROR: %s: %s(): line %d: unreachable code\n\n", file_name, func_name, line_num);

#if defined(__CUDA_ARCH__)
    asm("trap;");
#else
    exit(1);
#endif
}

} // namespace HIDDEN

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define CHECK_CUDA_ERROR(val) HIDDEN::_check_cuda_error((val), #val, __FILE__, __LINE__)

#define REPORT_FATAL_ERROR() HIDDEN::_report_error(__FILE__, __func__, __LINE__)

static constexpr bool DEBUG_MODE = false;
