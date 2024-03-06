#include "pbrt/scene/builder.h"

using namespace std;

int main(int argc, const char **argv) {
    {
        int runtime_version;
        cudaRuntimeGetVersion(&runtime_version);

        int major = runtime_version / 1000;
        int minor = runtime_version % 1000 / 10;
        int patch = runtime_version % 10;

        /*
        cudaDeviceSetLimit(cudaLimitMallocHeapSize,
        sizeof(RGBtoSpectrumData::RGBtoSpectrumTableGPU)); cudaDeviceSetLimit(cudaLimitStackSize,
        1024 * 8);
        */

        size_t heap_size;
        cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);

        size_t stack_size;
        cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);

        printf("CUDA info:\n");
        printf("    runtime version: %d.%d.%d\n", major, minor, patch);
        printf("    max stack size:  %zu\n", stack_size);
        printf("    max heap size:   %zu\n", heap_size);
        printf("\n");
        fflush(stdout);
    }

    const auto command_line_option = CommandLineOption(argc, argv);
    SceneBuilder::render_pbrt(command_line_option);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cudaDeviceReset();

    return 0;
}
