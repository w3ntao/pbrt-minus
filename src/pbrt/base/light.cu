#include "pbrt/base/light.h"

#include "pbrt/lights/diffuse_area_light.h"
#include "pbrt/lights/distant_light.h"
#include "pbrt/lights/image_infinite_light.h"

Light *Light::create(const std::string &type_of_light, const Transform &renderFromLight,
                     const ParameterDictionary &parameters,
                     std::vector<void *> &gpu_dynamic_pointers) {
    if (type_of_light == "distant") {
        auto distant_light =
            DistantLight::create(renderFromLight, parameters, gpu_dynamic_pointers);

        Light *light;
        CHECK_CUDA_ERROR(cudaMallocManaged(&light, sizeof(Light)));
        gpu_dynamic_pointers.push_back(light);

        light->init(distant_light);
        return light;
    }

    if (type_of_light == "infinite") {
        auto image_infinite_light =
            ImageInfiniteLight::create(renderFromLight, parameters, gpu_dynamic_pointers);

        Light *light;
        CHECK_CUDA_ERROR(cudaMallocManaged(&light, sizeof(Light)));
        gpu_dynamic_pointers.push_back(light);

        light->init(image_infinite_light);
        return light;
    }

    printf("\nLight `%s` not implemented\n", type_of_light.c_str());
    REPORT_FATAL_ERROR();
    return nullptr;
}

Light *Light::create_diffuse_area_light(const Shape *_shape, const Transform &render_from_light,
                                        const ParameterDictionary &parameters,
                                        std::vector<void *> &gpu_dynamic_pointers) {

    DiffuseAreaLight *diffuse_are_light;
    CHECK_CUDA_ERROR(cudaMallocManaged(&diffuse_are_light, sizeof(DiffuseAreaLight)));
    Light *light;
    CHECK_CUDA_ERROR(cudaMallocManaged(&light, sizeof(Light)));

    gpu_dynamic_pointers.push_back(diffuse_are_light);
    gpu_dynamic_pointers.push_back(light);

    diffuse_are_light->init(_shape, render_from_light, parameters);
    light->init(diffuse_are_light);

    return light;
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
LightType Light::get_light_type() const {
    switch (type) {
    case (Type::diffuse_area_light): {
        return ((DiffuseAreaLight *)ptr)->get_light_type();
    }

    case (Type::distant_light): {
        return ((DistantLight *)ptr)->get_light_type();
    }

    case (Type::image_infinite_light): {
        return ((ImageInfiniteLight *)ptr)->get_light_type();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_GPU
SampledSpectrum Light::l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                         const SampledWavelengths &lambda) const {
    switch (type) {
    case (Type::diffuse_area_light): {
        return ((DiffuseAreaLight *)ptr)->l(p, n, uv, w, lambda);
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_GPU
SampledSpectrum Light::le(const Ray &ray, const SampledWavelengths &lambda) const {
    switch (type) {
    case (Type::image_infinite_light): {
        return ((ImageInfiniteLight *)ptr)->le(ray, lambda);
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_GPU
cuda::std::optional<LightLiSample> Light::sample_li(const LightSampleContext ctx, const Point2f u,
                                                    SampledWavelengths &lambda) const {
    switch (type) {
    case (Type::diffuse_area_light): {
        return ((DiffuseAreaLight *)ptr)->sample_li(ctx, u, lambda);
    }

    case (Type::distant_light): {
        return ((DistantLight *)ptr)->sample_li(ctx, u, lambda);
    }

    case (Type::image_infinite_light): {
        return ((ImageInfiniteLight *)ptr)->sample_li(ctx, u, lambda);
    }
    }

    REPORT_FATAL_ERROR();
}

void Light::preprocess(const Bounds3<FloatType> &scene_bounds) {
    switch (type) {
    case (Type::diffuse_area_light): {
        // do nothing
        return;
    }

    case (Type::distant_light): {
        ((DistantLight *)ptr)->preprocess(scene_bounds);
        return;
    }

    case (Type::image_infinite_light): {
        ((ImageInfiniteLight *)ptr)->preprocess(scene_bounds);
        return;
    }
    }

    REPORT_FATAL_ERROR();
}
