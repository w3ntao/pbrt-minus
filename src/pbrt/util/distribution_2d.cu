#include "pbrt/util/distribution_2d.h"

#include "pbrt/spectrum_util/rgb.h"
#include "pbrt/util/distribution_1d.h"
#include "pbrt/textures/gpu_image.h"

const Distribution2D *Distribution2D::create_from_image(const GPUImage *image,
                                                        std::vector<void *> &gpu_dynamic_pointers) {
    Distribution2D *distribution;
    CHECK_CUDA_ERROR(cudaMallocManaged(&distribution, sizeof(Distribution2D)));
    gpu_dynamic_pointers.push_back(distribution);

    distribution->build_from_image(image, gpu_dynamic_pointers);

    return distribution;
}

void Distribution2D::build_from_image(const GPUImage *image,
                                      std::vector<void *> &gpu_dynamic_pointers) {
    dimension = image->resolution;

    distribution_1d_list = nullptr;
    cdf = nullptr;
    pmf = nullptr;

    FloatType *_pmf;
    CHECK_CUDA_ERROR(cudaMallocManaged(&_pmf, sizeof(FloatType) * dimension.x));
    gpu_dynamic_pointers.push_back(_pmf);

    double sum_pmf = 0.0;
    for (int x = 0; x < dimension.x; ++x) {
        FloatType accumulated_rgb_val = 0.0;
        for (int y = 0; y < dimension.y; ++y) {
            const auto rgb =
                image->fetch_pixel(Point2i(x, y), WrapMode::OctahedralSphere).clamp(0, Infinity);

            auto pixel_val = rgb.avg();
            accumulated_rgb_val += pixel_val;
        }

        _pmf[x] = accumulated_rgb_val;
        sum_pmf += accumulated_rgb_val;
    }

    for (uint idx = 0; idx < dimension.x; ++idx) {
        _pmf[idx] = _pmf[idx] / sum_pmf;
    }

    FloatType *_cdf;
    CHECK_CUDA_ERROR(cudaMallocManaged(&_cdf, sizeof(FloatType) * dimension.x));
    gpu_dynamic_pointers.push_back(_cdf);

    _cdf[0] = _pmf[0];
    for (uint idx = 1; idx < dimension.x; ++idx) {
        _cdf[idx] = _cdf[idx - 1] + _pmf[idx];
    }

    pmf = _pmf;
    cdf = _cdf;

    Distribution1D *_distribution_1d_list;
    CHECK_CUDA_ERROR(
        cudaMallocManaged(&_distribution_1d_list, sizeof(Distribution1D) * dimension.x));
    gpu_dynamic_pointers.push_back(_distribution_1d_list);

    for (int x = 0; x < dimension.x; ++x) {
        std::vector<FloatType> pdfs(dimension.y);
        for (int y = 0; y < dimension.y; ++y) {
            const auto rgb =
                image->fetch_pixel(Point2i(x, y), WrapMode::OctahedralSphere).clamp(0, Infinity);
            pdfs[y] = rgb.avg();
        }

        _distribution_1d_list[x].build(pdfs, gpu_dynamic_pointers);
    }

    distribution_1d_list = _distribution_1d_list;
}

PBRT_GPU
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
