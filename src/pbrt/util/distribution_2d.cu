#include <pbrt/spectrum_util/rgb.h>
#include <pbrt/util/distribution_1d.h>
#include <pbrt/util/distribution_2d.h>

#include <pbrt/gpu/gpu_memory_allocator.h>

const Distribution2D *Distribution2D::create(const std::vector<std::vector<FloatType>> &data,
                                             GPUMemoryAllocator &allocator) {
    auto distribution = allocator.allocate<Distribution2D>();

    distribution->build(data, allocator);

    return distribution;
}

void Distribution2D::build(const std::vector<std::vector<FloatType>> &data,
                           GPUMemoryAllocator &allocator) {
    if (data.empty()) {
        REPORT_FATAL_ERROR();
    }

    dimension = Point2i(data.size(), data[0].size());

    distribution_1d_list = nullptr;
    cdf = nullptr;
    pmf = nullptr;

    auto _pmf = allocator.allocate<FloatType>(dimension.x);

    double sum_pmf = 0.0;
    for (int x = 0; x < dimension.x; ++x) {
        FloatType sum_per_row = 0.0;
        for (int y = 0; y < dimension.y; ++y) {
            sum_per_row += data[x][y];
        }

        _pmf[x] = sum_per_row;
        sum_pmf += sum_per_row;
    }

    for (uint idx = 0; idx < dimension.x; ++idx) {
        _pmf[idx] = _pmf[idx] / sum_pmf;
    }

    auto _cdf = allocator.allocate<FloatType>(dimension.x);

    _cdf[0] = _pmf[0];
    for (uint idx = 1; idx < dimension.x; ++idx) {
        _cdf[idx] = _cdf[idx - 1] + _pmf[idx];
    }

    pmf = _pmf;
    cdf = _cdf;

    auto _distribution_1d_list = allocator.allocate<Distribution1D>(dimension.x);

    for (int x = 0; x < dimension.x; ++x) {
        std::vector<FloatType> pdfs(dimension.y);
        for (int y = 0; y < dimension.y; ++y) {
            pdfs[y] = data[x][y];
        }

        _distribution_1d_list[x].build(pdfs, allocator);
    }

    distribution_1d_list = _distribution_1d_list;
}

PBRT_CPU_GPU
cuda::std::pair<Point2f, FloatType> Distribution2D::sample(const Point2f &uv) const {
    auto first_dim_idx = search_cdf(uv.x, cdf, dimension.x);

    auto first_pdf = pmf[first_dim_idx];

    auto second_dim_result = distribution_1d_list[first_dim_idx].sample(uv.y);

    auto pdf = first_pdf * second_dim_result.second;

    return {Point2f(FloatType(first_dim_idx) / FloatType(dimension.x),
                    FloatType(second_dim_result.first) / FloatType(dimension.y)),
            pdf};
}

PBRT_CPU_GPU
FloatType Distribution2D::get_pdf(const Point2f &u) const {
    auto first_dim_index = clamp<uint>(u.x * FloatType(dimension.x), 0, dimension.x - 1);
    auto first_dim_pdf = pmf[first_dim_index];

    auto second_dim_index = clamp<uint>(u.y * FloatType(dimension.y), 0, dimension.y - 1);
    auto second_dim_pdf = distribution_1d_list[first_dim_index].get_pdf(second_dim_index);

    return first_dim_pdf * second_dim_pdf;
}
