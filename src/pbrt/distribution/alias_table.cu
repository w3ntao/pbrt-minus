#include <functional>
#include <numeric>
#include <pbrt/distribution/alias_table.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/math.h>

const AliasTable *AliasTable::create(const std::vector<Real> &values,
                                     GPUMemoryAllocator &allocator) {
    if (values.empty()) {
        REPORT_FATAL_ERROR();
    }

    const auto size = values.size();

    auto bins_to_sort = std::vector<Bin>(size);
    std::vector<Bin> bins_done_sorting;

    const auto value_sum = std::accumulate(values.begin(), values.end(), 0.0);
    if (value_sum == 0) {
        auto gpu_bins = allocator.allocate<Bin>(size);
        auto gpu_pdfs = allocator.allocate<Real>(size);

        for (uint idx = 0; idx < size; ++idx) {
            gpu_bins[idx] = Bin(1.0, idx);
            gpu_pdfs[idx] = Real(idx) / size;
        }

        auto alias_table = allocator.allocate<AliasTable>();
        alias_table->size = size;
        alias_table->bins = gpu_bins;
        alias_table->pdfs = gpu_pdfs;

        return alias_table;
    }

    auto gpu_pdfs = allocator.allocate<Real>(size);
    for (uint idx = 0; idx < size; ++idx) {
        bins_to_sort[idx] = Bin(values[idx] * size / value_sum, idx);
        gpu_pdfs[idx] = values[idx] / value_sum;
    }

    constexpr Real tolerance = 0.001;

    while (true) {
        if (bins_to_sort.size() == 1) {
            bins_done_sorting.push_back(bins_to_sort[0]);
            bins_to_sort.clear();
            break;
        }

        std::sort(bins_to_sort.begin(), bins_to_sort.end(), std::less{});

        if (bins_to_sort[0].p < 1 - tolerance) {
            const auto underflow_prob = 1 - bins_to_sort[0].p;

            auto last_bin = &bins_to_sort[bins_to_sort.size() - 1];
            bins_to_sort[0].second_idx = last_bin->first_idx;
            last_bin->p -= underflow_prob;
        }

        bins_done_sorting.push_back(bins_to_sort[0]);
        bins_to_sort.erase(bins_to_sort.begin() + 0);
    }

    auto gpu_bins = allocator.allocate<Bin>(size);
    cudaMemcpy(gpu_bins, bins_done_sorting.data(), sizeof(Bin) * size, cudaMemcpyHostToDevice);

    auto alias_table = allocator.allocate<AliasTable>();

    alias_table->size = size;
    alias_table->bins = gpu_bins;
    alias_table->pdfs = gpu_pdfs;

    return alias_table;
}

PBRT_CPU_GPU
cuda::std::pair<uint, Real> AliasTable::sample(const Real u0) const {
    const auto idx = clamp<int>(u0 * double(size), 0, size - 1);
    if (bins[idx].second_idx < 0) {
        const auto sampled_idx = bins[idx].first_idx;
        return {sampled_idx, pdfs[sampled_idx]};
    }

    const auto u1 = pbrt::hash_float(u0, idx);

    const auto sampled_idx = u1 <= bins[idx].p ? bins[idx].first_idx : bins[idx].second_idx;

    if (DEBUG_MODE && sampled_idx < 0) {
        REPORT_FATAL_ERROR();
    }

    return {sampled_idx, pdfs[sampled_idx]};
}
