#include <pbrt/scene/scene_builder.h>
#include <iostream>

using namespace std;

// taken from https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h
inline int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128}, {0x53, 128},
        {0x60, 64},  {0x61, 128}, {0x62, 128}, {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},
        {0x86, 128}, {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf("MapSMtoCores for SM %d.%d is undefined."
           "  Default to use %d Cores/SM\n",
           major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

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

        auto cuda_cores = _ConvertSMVer2Cores(props.major, props.minor) * props.multiProcessorCount;

        printf("    device %d: %s\n", i, props.name);
        printf("        compute capability: %d.%d\n", props.major, props.minor);
        printf("        cuda cores:         %u\n", cuda_cores);
        printf("        total memory:       %.2f GB\n",
               float(props.totalGlobalMem / 1024 / 1024) / 1024.0f);
        printf("        max threads per block: %u\n", props.maxThreadsPerBlock);
        printf("        wrap size:             %u\n", props.warpSize);
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
    size_t stack_size;
    cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
    size_t new_stack_size = std::max(stack_size, size_t(64 * 1024));
    CHECK_CUDA_ERROR(cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size));

    display_system_info();

#ifdef PBRT_FLOAT_AS_DOUBLE
    std::cout << "Float type: double\n";
#else
    std::cout << "Float type: float\n";
#endif

    const auto command_line_option = CommandLineOption(argc, argv);
    SceneBuilder::render_pbrt(command_line_option);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaDeviceReset());

    return 0;
}
