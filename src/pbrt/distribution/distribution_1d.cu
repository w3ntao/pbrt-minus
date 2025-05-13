#include <pbrt/distribution/alias_table.h>
#include <pbrt/distribution/distribution_1d.h>
#include <pbrt/gpu/gpu_memory_allocator.h>

const Distribution1D *Distribution1D::create(const std::vector<Real> &values,
                                             GPUMemoryAllocator &allocator) {
    auto distribution_1D = allocator.allocate<Distribution1D>();
    distribution_1D->build(values, allocator);

    return distribution_1D;
}

void Distribution1D::build(const std::vector<Real> &values, GPUMemoryAllocator &allocator) {
    alias_table = AliasTable::create(values, allocator);
}

PBRT_CPU_GPU
cuda::std::pair<uint, Real> Distribution1D::sample(const Real u) const {
    return alias_table->sample(u);
}

PBRT_CPU_GPU
Real Distribution1D::get_pdf(const uint idx) const {
    return alias_table->pdfs[idx];
}
