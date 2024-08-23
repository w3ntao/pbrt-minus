#include "pbrt/lights/image_infinite_light.h"

#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/euclidean_space/vector3.h"

#include "pbrt/scene/parameter_dictionary.h"

#include "pbrt/spectra/rgb_illuminant_spectrum.h"

#include "pbrt/spectrum_util/global_spectra.h"
#include "pbrt/spectrum_util/rgb_color_space.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"

#include "pbrt/textures/gpu_image.h"

#include "pbrt/util/distribution_2d.h"
#include "pbrt/util/macro.h"
#include "pbrt/util/math.h"

ImageInfiniteLight *ImageInfiniteLight::create(const Transform &render_from_light,
                                               const ParameterDictionary &parameters,
                                               std::vector<void *> &gpu_dynamic_pointers) {
    ImageInfiniteLight *image_infinite_light;
    CHECK_CUDA_ERROR(cudaMallocManaged(&image_infinite_light, sizeof(ImageInfiniteLight)));
    gpu_dynamic_pointers.push_back(image_infinite_light);

    image_infinite_light->init(render_from_light, parameters, gpu_dynamic_pointers);

    return image_infinite_light;
}

void ImageInfiniteLight::init(const Transform &_render_from_light,
                              const ParameterDictionary &parameters,
                              std::vector<void *> &gpu_dynamic_pointers) {
    auto _scale = parameters.get_float("scale", 1.0);

    const auto cie_y = parameters.global_spectra->cie_y;

    _scale /= parameters.global_spectra->rgb_color_space->illuminant->to_photometric(cie_y);
    scale = _scale;

    light_type = LightType::infinite;
    render_from_light = _render_from_light;

    auto texture_file = parameters.root + "/" + parameters.get_one_string("filename");
    image_ptr = GPUImage::create_from_file(texture_file, gpu_dynamic_pointers);

    image_resolution = image_ptr->get_resolution();

    image_le_distribution = Distribution2D::create_from_image(image_ptr, gpu_dynamic_pointers);

    color_space = parameters.global_spectra->rgb_color_space;

    scene_radius = NAN;
    scene_center = Point3f(NAN, NAN, NAN);
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
    const auto le_sample = image_le_distribution->sample(u);

    const auto uv = le_sample.first;
    const auto map_pdf = le_sample.second * image_resolution.x * image_resolution.y;

    Vector3f wLight = EqualAreaSquareToSphere(uv);
    Vector3f wi = render_from_light(wLight);

    // Compute PDF for sampled infinite light direction
    FloatType pdf = map_pdf / (4 * compute_pi());

    const auto interaction = Interaction(ctx.p() + wi * (2 * scene_radius));

    // Return radiance value for infinite light direction
    return LightLiSample(ImageLe(uv, lambda), wi, pdf, interaction);
}

PBRT_CPU_GPU SampledSpectrum ImageInfiniteLight::phi(const SampledWavelengths &lambda) const {
    SampledSpectrum sumL(0.0);

    auto width = image_resolution.x;
    auto height = image_resolution.y;

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            auto rgb = image_ptr->fetch_pixel(Point2i(x, y), WrapMode::OctahedralSphere);
            sumL += RGBIlluminantSpectrum(rgb.clamp(0.0, Infinity), color_space).sample(lambda);
        }
    }

    const auto pi = compute_pi();

    // Integrating over the sphere, so 4pi for that.  Then one more for Pi
    // r^2 for the area of the disk receiving illumination...
    return 4 * pi * pi * sqr(scene_radius) * scale * sumL / (width * height);
}

PBRT_GPU
FloatType ImageInfiniteLight::pdf_li(const LightSampleContext &ctx, const Vector3f &w,
                                     bool allow_incomplete_pdf) const {
    Vector3f wLight = render_from_light.apply_inverse(w);
    Point2f uv = EqualAreaSphereToSquare(wLight);

    auto pdf = this->image_le_distribution->get_pdf(uv) * image_resolution.x * image_resolution.y;
    return pdf / (4 * compute_pi());
}

void ImageInfiniteLight::preprocess(const Bounds3f &scene_bounds) {
    scene_bounds.bounding_sphere(&scene_center, &scene_radius);
}

PBRT_GPU
SampledSpectrum ImageInfiniteLight::ImageLe(Point2f uv, const SampledWavelengths &lambda) const {
    const auto rgb = image_ptr->bilerp(uv, WrapMode::OctahedralSphere).clamp(0, Infinity);

    auto spec = RGBIlluminantSpectrum(rgb, color_space);

    return scale * spec.sample(lambda);
}
