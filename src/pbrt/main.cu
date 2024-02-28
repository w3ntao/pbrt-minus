#include "pbrt/scene/builder.h"

using namespace std;

int main(int argc, const char **argv) {
    /*
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeof(RGBtoSpectrumData::RGBtoSpectrumTableGPU));
    cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 8);

    size_t heapSize;
    cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);

    size_t stackSize;
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);

    printf("max stack size: %d\n", stackSize);
    printf("max heap size:  %d\n", heapSize);
    */

    const auto command_line_option = CommandLineOption(argc, argv);
    SceneBuilder::render_pbrt(command_line_option);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cudaDeviceReset();

    return 0;
}
