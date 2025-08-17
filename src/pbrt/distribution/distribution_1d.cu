#include <pbrt/distribution/alias_table.h>
#include <pbrt/distribution/distribution_1d.h>
#include <pbrt/gpu/gpu_memory_allocator.h>

Distribution1D::Distribution1D(const std::vector<Real> &values, GPUMemoryAllocator &allocator)
    : alias_table(allocator.create<AliasTable>(values, allocator)) {}

PBRT_CPU_GPU
cuda::std::pair<int, Real> Distribution1D::sample(const Real u) const {
    return alias_table->sample(u);
}

PBRT_CPU_GPU
Real Distribution1D::get_pdf(const int idx) const {
    return alias_table->pdfs[idx];
}
