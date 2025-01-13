#include <pbrt/base/light.h>
#include <pbrt/base/shape.h>
#include <pbrt/lights/diffuse_area_light.h>
#include <pbrt/lights/distant_light.h>
#include <pbrt/lights/image_infinite_light.h>
#include <pbrt/lights/spot_light.h>
#include <pbrt/lights/uniform_infinite_light.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/global_spectra.h>

#include <pbrt/gpu/gpu_memory_allocator.h>

template <typename TypeOfLight>
static __global__ void init_lights(Light *lights, TypeOfLight *concrete_lights, uint num) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num) {
        return;
    }

    lights[worker_idx].init(&concrete_lights[worker_idx]);
}

Light *Light::create(const std::string &type_of_light, const Transform &render_from_light,
                     const ParameterDictionary &parameters, GPUMemoryAllocator &allocator) {
    auto light = allocator.allocate<Light>();

    if (type_of_light == "distant") {
        auto distant_light = DistantLight::create(render_from_light, parameters, allocator);

        light->init(distant_light);

        return light;
    }

    if (type_of_light == "infinite") {
        if (parameters.has_string("filename")) {
            auto image_infinite_light =
                ImageInfiniteLight::create(render_from_light, parameters, allocator);

            light->init(image_infinite_light);

            return light;
        }
        // otherwise it's UniformInfiniteLight

        auto uniform_infinite_light =
            UniformInfiniteLight::create(render_from_light, parameters, allocator);

        light->init(uniform_infinite_light);

        return light;
    }

    if (type_of_light == "spot") {
        auto spot_light = SpotLight::create(render_from_light, parameters, allocator);

        light->init(spot_light);
        return light;
    }

    printf("\n%s(): Light `%s` not implemented\n", __func__, type_of_light.c_str());
    REPORT_FATAL_ERROR();
    return nullptr;
}

Light *Light::create_diffuse_area_lights(const Shape *shapes, const uint num,
                                         const Transform &render_from_light,
                                         const ParameterDictionary &parameters,
                                         GPUMemoryAllocator &allocator) {
    constexpr uint threads = 1024;
    const uint blocks = divide_and_ceil(num, threads);

    auto diffuse_area_lights = allocator.allocate<DiffuseAreaLight>(num);
    auto lights = allocator.allocate<Light>(num);

    for (uint idx = 0; idx < num; idx++) {
        diffuse_area_lights[idx].init(&shapes[idx], render_from_light, parameters, allocator);
    }

    init_lights<<<blocks, threads>>>(lights, diffuse_area_lights, num);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return lights;
}

PBRT_CPU_GPU
void Light::init(DistantLight *distant_light) {
    type = Type::distant_light;
    ptr = distant_light;
}

PBRT_CPU_GPU
void Light::init(DiffuseAreaLight *diffuse_area_light) {
    type = Type::diffuse_area_light;
    ptr = diffuse_area_light;
}

PBRT_CPU_GPU
void Light::init(ImageInfiniteLight *image_infinite_light) {
    type = Type::image_infinite_light;
    ptr = image_infinite_light;
}

PBRT_CPU_GPU
void Light::init(SpotLight *spot_light) {
    type = Type::spot_light;
    ptr = spot_light;
}

PBRT_CPU_GPU
void Light::init(UniformInfiniteLight *uniform_infinite_light) {
    type = Type::uniform_infinite_light;
    ptr = uniform_infinite_light;
}

PBRT_CPU_GPU
LightType Light::get_light_type() const {
    switch (type) {
    case Type::diffuse_area_light: {
        return static_cast<const DiffuseAreaLight *>(ptr)->get_light_type();
    }

    case Type::distant_light: {
        return static_cast<const DistantLight *>(ptr)->get_light_type();
    }

    case Type::image_infinite_light: {
        return static_cast<const ImageInfiniteLight *>(ptr)->get_light_type();
    }

    case Type::spot_light: {
        return static_cast<const SpotLight *>(ptr)->get_light_type();
    }

    case Type::uniform_infinite_light: {
        return static_cast<const UniformInfiniteLight *>(ptr)->get_light_type();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
SampledSpectrum Light::l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                         const SampledWavelengths &lambda) const {
    switch (type) {
    case Type::diffuse_area_light: {
        return static_cast<const DiffuseAreaLight *>(ptr)->l(p, n, uv, w, lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
SampledSpectrum Light::le(const Ray &ray, const SampledWavelengths &lambda) const {
    switch (type) {
    case Type::image_infinite_light: {
        return static_cast<const ImageInfiniteLight *>(ptr)->le(ray, lambda);
    }

    case Type::uniform_infinite_light: {
        return static_cast<const UniformInfiniteLight *>(ptr)->le(ray, lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
pbrt::optional<LightLiSample> Light::sample_li(const LightSampleContext &ctx, const Point2f &u,
                                               SampledWavelengths &lambda) const {
    switch (type) {
    case Type::diffuse_area_light: {
        return static_cast<const DiffuseAreaLight *>(ptr)->sample_li(ctx, u, lambda);
    }

    case Type::distant_light: {
        return static_cast<const DistantLight *>(ptr)->sample_li(ctx, u, lambda);
    }

    case Type::image_infinite_light: {
        return static_cast<const ImageInfiniteLight *>(ptr)->sample_li(ctx, u, lambda);
    }

    case Type::spot_light: {
        return static_cast<const SpotLight *>(ptr)->sample_li(ctx, u, lambda);
    }

    case Type::uniform_infinite_light: {
        return static_cast<const UniformInfiniteLight *>(ptr)->sample_li(ctx, u, lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
pbrt::optional<LightLeSample> Light::sample_le(const Point2f u1, const Point2f u2,
                                               SampledWavelengths &lambda) const {
    switch (type) {
    case Type::spot_light: {
        return static_cast<const SpotLight *>(ptr)->sample_le(u1, u2, lambda);
    }

    case Type::uniform_infinite_light: {
        return static_cast<const UniformInfiniteLight *>(ptr)->sample_le(u1, u2, lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
FloatType Light::pdf_li(const LightSampleContext &ctx, const Vector3f &wi,
                        bool allow_incomplete_pdf) const {
    switch (type) {
    case Type::diffuse_area_light: {
        return static_cast<const DiffuseAreaLight *>(ptr)->pdf_li(ctx, wi, allow_incomplete_pdf);
    }

    case Type::image_infinite_light: {
        return static_cast<const ImageInfiniteLight *>(ptr)->pdf_li(ctx, wi, allow_incomplete_pdf);
    }

    case Type::spot_light: {
        return static_cast<const SpotLight *>(ptr)->pdf_li(ctx, wi, allow_incomplete_pdf);
    }

    case Type::uniform_infinite_light: {
        return static_cast<const UniformInfiniteLight *>(ptr)->pdf_li(ctx, wi,
                                                                      allow_incomplete_pdf);
    }
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

PBRT_CPU_GPU
void Light::pdf_le(const Interaction &intr, Vector3f w, FloatType *pdfPos,
                   FloatType *pdfDir) const {
    switch (type) {
    case Type::spot_light: {
        printf("ERROR: you shouldn't call %s() from non-area lights\n", __func__);
        REPORT_FATAL_ERROR();
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_CPU_GPU
void Light::pdf_le(const Ray &ray, FloatType *pdfPos, FloatType *pdfDir) const {
    switch (type) {
    case Type::spot_light: {
        static_cast<const SpotLight *>(ptr)->pdf_le(ray, pdfPos, pdfDir);
        return;
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_CPU_GPU
SampledSpectrum Light::phi(const SampledWavelengths &lambda) const {
    switch (type) {
    case Type::diffuse_area_light: {
        return static_cast<const DiffuseAreaLight *>(ptr)->phi(lambda);
    }

    case Type::distant_light: {
        return static_cast<const DistantLight *>(ptr)->phi(lambda);
    }

    case Type::image_infinite_light: {
        return static_cast<const ImageInfiniteLight *>(ptr)->phi(lambda);
    }

    case Type::spot_light: {
        return static_cast<const SpotLight *>(ptr)->phi(lambda);
    }

    case Type::uniform_infinite_light: {
        return static_cast<const UniformInfiniteLight *>(ptr)->phi(lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

void Light::preprocess(const Bounds3<FloatType> &scene_bounds) {
    switch (type) {
    case Type::diffuse_area_light: {
        // do nothing
        return;
    }

    case Type::spot_light: {
        // do nothing
        return;
    }

    case Type::distant_light: {
        static_cast<DistantLight *>(ptr)->preprocess(scene_bounds);
        return;
    }

    case Type::image_infinite_light: {
        static_cast<ImageInfiniteLight *>(ptr)->preprocess(scene_bounds);
        return;
    }

    case Type::uniform_infinite_light: {
        static_cast<UniformInfiniteLight *>(ptr)->preprocess(scene_bounds);
        return;
    }
    }

    REPORT_FATAL_ERROR();
}
