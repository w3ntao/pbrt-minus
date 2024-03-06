#include "pbrt/scene/builder.h"

#include <algorithm>
#include <random>

using namespace std;

void display_system_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("CUDA devices: %d\n", device_count);
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        printf("    %d -- %s: %d.%d\n", i, props.name, props.major, props.minor);
    }
    int runtime_version;
    cudaRuntimeGetVersion(&runtime_version);

    int major = runtime_version / 1000;
    int minor = runtime_version % 1000 / 10;
    int patch = runtime_version % 10;

    size_t heap_size;
    cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);

    size_t stack_size;
    cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);

    printf("    runtime version: %d.%d.%d\n", major, minor, patch);
    printf("    stack size limit:  %zu KB\n", stack_size / 1024);
    printf("    heap size limit:   %zu KB\n", heap_size / 1024);
    printf("\n");
    fflush(stdout);
}

int main(int argc, const char **argv) {
    {
        size_t stack_size;
        cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
        size_t new_stack_size = std::max(stack_size, size_t(8 * 1024));
        checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size));

        // TODO: progress 2024/03/08 you need this much (128MB) heap space to get killeroo working
        checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024));
    }

    display_system_info();

    const auto command_line_option = CommandLineOption(argc, argv);
    SceneBuilder::render_pbrt(command_line_option);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cudaDeviceReset();

    return 0;
}
