#include "pbrt/scene/builder.h"

#include <algorithm>
#include <random>

using namespace std;

void print_bytes(size_t bytes) {
    size_t giga = 1024 * 1024 * 1024;
    size_t mega = 1024 * 1024;
    size_t kilo = 1024;

    if (bytes >= giga) {
        printf("%zu GB", bytes / giga);
        return;
    }

    if (bytes >= mega) {
        printf("%zu MB", bytes / mega);
        return;
    }

    if (bytes >= kilo) {
        printf("%zu KB", bytes / kilo);
        return;
    }

    printf("%zu B", bytes);
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
__device__ void print_arch_device() {
    constexpr char compile_time_arch[] = STR(__CUDA_ARCH__);
    printf("    SM arch: %s\n", compile_time_arch);
}
#undef STR
#undef STR_HELPER

__global__ void print_arch_global() {
    print_arch_device();
}

void display_system_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("CUDA devices: %d\n", device_count);
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        printf("    device %d: %s\n", i, props.name);
        printf("        compute capability: %d.%d\n", props.major, props.minor);
        printf("        max threads per block: %u\n", props.maxThreadsPerBlock);
    }

    print_arch_global<<<1, 1>>>();
    cudaDeviceSynchronize();

    int runtime_version;
    cudaRuntimeGetVersion(&runtime_version);

    int major = runtime_version / 1000;
    int minor = runtime_version % 1000 / 10;
    int patch = runtime_version % 10;

    printf("    runtime: %d.%d.%d\n", major, minor, patch);

    size_t heap_size;
    cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);

    size_t stack_size;
    cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);

    printf("    stack size limit: ");
    print_bytes(stack_size);
    printf("\n");

    printf("    heap size limit:  ");
    print_bytes(heap_size);
    printf("\n");

    printf("\n");
    fflush(stdout);
}

int main(int argc, const char **argv) {
    {
        size_t stack_size;
        cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
        size_t new_stack_size = std::max(stack_size, size_t(8 * 1024));
        checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size));
    }

    display_system_info();

    const auto command_line_option = CommandLineOption(argc, argv);
    SceneBuilder::render_pbrt(command_line_option);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cudaDeviceReset();

    return 0;
}
