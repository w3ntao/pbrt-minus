#include <pbrt/distribution/alias_table.h>
#include <pbrt/distribution/distribution_1d.h>
#include <pbrt/distribution/distribution_2d.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/spectrum_util/rgb.h>
#include <pbrt/util/thread_pool.h>

Distribution2D::Distribution2D(const std::vector<std::vector<Real>> &data,
                               GPUMemoryAllocator &allocator) {
    if (data.empty()) {
        REPORT_FATAL_ERROR();
    }

    dimension = Point2i(data.size(), data[0].size());

    std::vector<Real> sum_per_row(dimension.x, 0);
    for (int x = 0; x < dimension.x; ++x) {
        Real current_sum = 0.0;
        for (int y = 0; y < dimension.y; ++y) {
            current_sum += data[x][y];
        }

        sum_per_row[x] = current_sum;
    }

    dimension_x_distribution = allocator.create<AliasTable>(sum_per_row, allocator);

    auto _distribution_1d_list = allocator.allocate<Distribution1D>(dimension.x);

    // not an elegant solution indeed, but it's simple enough
    ThreadPool thread_pool;
    thread_pool.parallel_execute(
        0, dimension.x,
        [data, &_distribution_1d_list, &allocator, dimension = this->dimension](const int x) {
            std::vector<Real> pdfs(dimension.y);
            for (int y = 0; y < dimension.y; ++y) {
                pdfs[y] = data[x][y];
            }

            _distribution_1d_list[x] = Distribution1D(pdfs, allocator);
        });

    dimension_y_distribution_list = _distribution_1d_list;
}

PBRT_CPU_GPU
cuda::std::pair<Point2f, Real> Distribution2D::sample(const Point2f &uv) const {
    auto [sampled_dim_x, first_pdf] = dimension_x_distribution->sample(uv.x);
    sampled_dim_x = clamp<int>(sampled_dim_x, 0, dimension.x - 1);

    auto [sampled_dim_y, second_pdf] = dimension_y_distribution_list[sampled_dim_x].sample(uv.y);

    return {
        Point2f(Real(sampled_dim_x) / Real(dimension.x), Real(sampled_dim_y) / Real(dimension.y)),
        first_pdf * second_pdf};
}

PBRT_CPU_GPU
Real Distribution2D::get_pdf(const Point2f &u) const {
    const auto first_dimension_index = clamp<int>(u.x * Real(dimension.x), 0, dimension.x - 1);
    const auto first_dimension_pdf = dimension_x_distribution->pdfs[first_dimension_index];

    const auto second_dimension_index = clamp<int>(u.y * Real(dimension.y), 0, dimension.y - 1);
    const auto second_dimension_pdf =
        dimension_y_distribution_list[first_dimension_index].get_pdf(second_dimension_index);

    return first_dimension_pdf * second_dimension_pdf;
}
