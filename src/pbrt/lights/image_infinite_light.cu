#include "pbrt/lights/image_infinite_light.h"

#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/euclidean_space/vector3.h"

#include "pbrt/scene/parameter_dictionary.h"

#include "pbrt/spectra/rgb_illuminant_spectrum.h"

#include "pbrt/spectrum_util/global_spectra.h"
#include "pbrt/spectrum_util/rgb_color_space.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"

#include "pbrt/textures/gpu_image.h"
#include "pbrt/util/macro.h"
#include "pbrt/util/math.h"

ImageInfiniteLight *ImageInfiniteLight::create(const Transform &_render_from_light,
                                               const ParameterDictionary &parameters,
                                               std::vector<void *> &gpu_dynamic_pointers) {

    auto texture_file = parameters.root + "/" + parameters.get_string("filename", std::nullopt);

    auto scale = parameters.get_float("scale", 1.0);

    const auto cie_y = parameters.global_spectra->cie_y;

    scale /= parameters.global_spectra->rgb_color_space->illuminant->to_photometric(cie_y);

    ImageInfiniteLight *image_infinite_light;
    CHECK_CUDA_ERROR(cudaMallocManaged(&image_infinite_light, sizeof(ImageInfiniteLight)));
    gpu_dynamic_pointers.push_back(image_infinite_light);

    image_infinite_light->light_type = LightType::infinite;
    image_infinite_light->render_from_light = _render_from_light;

    image_infinite_light->image = GPUImage::create_from_file(texture_file, gpu_dynamic_pointers);
    image_infinite_light->scale = scale;

    image_infinite_light->color_space = parameters.global_spectra->rgb_color_space;

    image_infinite_light->scene_radius = NAN;
    image_infinite_light->scene_center = Point3f(NAN, NAN, NAN);

    return image_infinite_light;
}

PBRT_GPU
SampledSpectrum ImageInfiniteLight::le(const Ray &ray, const SampledWavelengths &lambda) const {
    Vector3f wLight = (render_from_light.apply_inverse(ray.d)).normalize();
    auto uv = EqualAreaSphereToSquare(wLight);
    return ImageLe(uv, lambda);
}

PBRT_GPU
cuda::std::optional<LightLiSample> ImageInfiniteLight::sample_li(const LightSampleContext &ctx,
                                                                 const Point2f &u,
                                                                 SampledWavelengths &lambda) const {
    // Convert infinite light sample point to direction
    Vector3f wLight = EqualAreaSquareToSphere(u);
    Vector3f wi = render_from_light(wLight);

    // Compute PDF for sampled infinite light direction
    FloatType pdf = 1.0 / (4 * compute_pi());

    const auto interaction = Interaction(ctx.p() + wi * (2 * scene_radius));

    // Return radiance value for infinite light direction
    return LightLiSample(ImageLe(u, lambda), wi, pdf, interaction);
}

PBRT_CPU_GPU
SampledSpectrum ImageInfiniteLight::phi(const SampledWavelengths &lambda) const {
    SampledSpectrum sumL(0.0);

    auto width = image->resolution.x;
    auto height = image->resolution.y;

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            auto rgb = image->fetch_pixel(Point2i(u, v), WrapMode::OctahedralSphere);
            sumL += RGBIlluminantSpectrum(rgb.clamp(0.0, Infinity), color_space).sample(lambda);
        }
    }

    const auto pi = compute_pi();

    // Integrating over the sphere, so 4pi for that.  Then one more for Pi
    // r^2 for the area of the disk receiving illumination...
    return 4 * pi * pi * sqr(scene_radius) * scale * sumL / (width * height);
}

PBRT_GPU
FloatType ImageInfiniteLight::pdf_li(const LightSampleContext &ctx, const Vector3f &wi,
                                     bool allow_incomplete_pdf) const {
    // allow_incomplete_pdf = false
    return 1.0 / (4 * compute_pi());
}

void ImageInfiniteLight::preprocess(const Bounds3f &scene_bounds) {
    scene_bounds.bounding_sphere(&scene_center, &scene_radius);
}

PBRT_GPU
SampledSpectrum ImageInfiniteLight::ImageLe(Point2f uv, const SampledWavelengths &lambda) const {
    const auto rgb = image->bilerp(uv, WrapMode::OctahedralSphere).clamp(0, Infinity);

    auto spec = RGBIlluminantSpectrum(rgb, color_space);

    return scale * spec.sample(lambda);
}
