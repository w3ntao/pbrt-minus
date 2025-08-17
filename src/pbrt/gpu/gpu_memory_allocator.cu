#include <iomanip>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/util/math.h>

std::string GPUMemoryAllocator::get_allocated_memory_size() const {
    size_t allocated_memory = 0;
    for (const auto size : allocated_pointers | std::views::values) {
        allocated_memory += size;
    }

    const auto size_in_mb = divide_and_ceil<ulong>(allocated_memory, 1024 * 1024);

    if (size_in_mb < 1024) {
        return std::to_string(size_in_mb) + " MB";
    }

    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << (Real(size_in_mb) / 1024);
    return stream.str() + " GB";
}
