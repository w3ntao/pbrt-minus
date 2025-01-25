#include <pbrt/distribution/alias_table.h>
#include <pbrt/distribution/distribution_1d.h>
#include <pbrt/distribution/distribution_2d.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/spectrum_util/rgb.h>
#include <pbrt/util/thread_pool.h>

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

    std::vector<FloatType> sum_per_row(dimension.x, 0);
    for (int x = 0; x < dimension.x; ++x) {
        FloatType current_sum = 0.0;
        for (int y = 0; y < dimension.y; ++y) {
            current_sum += data[x][y];
        }

        sum_per_row[x] = current_sum;
    }

    dimension_x_distribution = AliasTable::create(sum_per_row, allocator);

    auto _distribution_1d_list = allocator.allocate<Distribution1D>(dimension.x);

    // not an elegant solution indeed, but it's simple enough
    ThreadPool thread_pool;
    thread_pool.parallel_execute(
        0, dimension.x,
        [data, &_distribution_1d_list, &allocator, dimension = this->dimension](const int x) {
            std::vector<FloatType> pdfs(dimension.y);
            for (int y = 0; y < dimension.y; ++y) {
                pdfs[y] = data[x][y];
            }

            _distribution_1d_list[x].build(pdfs, allocator);
        });

    dimension_y_distribution_list = _distribution_1d_list;
}

PBRT_CPU_GPU
cuda::std::pair<Point2f, FloatType> Distribution2D::sample(const Point2f &uv) const {
    auto first_dim_sample = dimension_x_distribution->sample(uv.x);

    auto sampled_dim_x = clamp<int>(first_dim_sample.first, 0, dimension.x - 1);

    auto first_pdf = first_dim_sample.second;

    auto second_dim_sample = dimension_y_distribution_list[sampled_dim_x].sample(uv.y);

    auto pdf = first_pdf * second_dim_sample.second;

    return {Point2f(FloatType(sampled_dim_x) / FloatType(dimension.x),
                    FloatType(second_dim_sample.first) / FloatType(dimension.y)),
            pdf};
}

PBRT_CPU_GPU
FloatType Distribution2D::get_pdf(const Point2f &u) const {
    const auto first_dimension_index =
        clamp<uint>(u.x * FloatType(dimension.x), 0, dimension.x - 1);
    const auto first_dimension_pdf = dimension_x_distribution->pdfs[first_dimension_index];

    const auto second_dimension_index =
        clamp<uint>(u.y * FloatType(dimension.y), 0, dimension.y - 1);
    const auto second_dimension_pdf =
        dimension_y_distribution_list[first_dimension_index].get_pdf(second_dimension_index);

    return first_dimension_pdf * second_dimension_pdf;
}
