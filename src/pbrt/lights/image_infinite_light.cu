#include <pbrt/distribution/distribution_2d.h>
#include <pbrt/euclidean_space/bounds3.h>
#include <pbrt/euclidean_space/vector3.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/gpu/macro.h>
#include <pbrt/lights/image_infinite_light.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectra/rgb_illuminant_spectrum.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/spectrum_util/rgb_color_space.h>
#include <pbrt/spectrum_util/sampled_spectrum.h>
#include <pbrt/textures/gpu_image.h>
#include <pbrt/util/sampling.h>

ImageInfiniteLight *ImageInfiniteLight::create(const Transform &render_from_light,
                                               const ParameterDictionary &parameters,
                                               GPUMemoryAllocator &allocator) {
    auto image_infinite_light = allocator.allocate<ImageInfiniteLight>();

    image_infinite_light->init(render_from_light, parameters, allocator);

    return image_infinite_light;
}

void ImageInfiniteLight::init(const Transform &_render_from_light,
                              const ParameterDictionary &parameters,
                              GPUMemoryAllocator &allocator) {
    auto _scale = parameters.get_float("scale", 1.0);

    const auto cie_y = parameters.global_spectra->cie_y;

    _scale /= parameters.global_spectra->rgb_color_space->illuminant->to_photometric(cie_y);
    scale = _scale;

    light_type = LightType::infinite;
    render_from_light = _render_from_light;

    color_space = parameters.global_spectra->rgb_color_space;

    scene_radius = NAN;
    scene_center = Point3f(NAN, NAN, NAN);

    auto texture_file = parameters.root + "/" + parameters.get_one_string("filename");
    image_ptr = GPUImage::create_from_file(texture_file, allocator);

    image_resolution = image_ptr->get_resolution();

    Real max_luminance = 0.0;
    std::vector<std::vector<Real>> image_luminance_array(image_resolution.x,
                                                         std::vector<Real>(image_resolution.y));
    for (int x = 0; x < image_resolution.x; ++x) {
        for (int y = 0; y < image_resolution.y; ++y) {
            const auto rgb = image_ptr->fetch_pixel(Point2i(x, y), WrapMode::OctahedralSphere)
                                 .clamp(0, Infinity);

            auto luminance = rgb.avg();
            image_luminance_array[x][y] = luminance;

            max_luminance = std::max(max_luminance, luminance);
        }
    }

    if (max_luminance <= 0.0) {
        REPORT_FATAL_ERROR();
    }

    // ignore minimal values
    // those pixels with luminance smaller than 0.001 * max_luminance are ignored
    const auto ignore_ratio = 0.001;
    auto num_ignore = 0;
    auto ignore_threshold = ignore_ratio * max_luminance;
    for (int x = 0; x < image_resolution.x; ++x) {
        for (int y = 0; y < image_resolution.y; ++y) {
            if (image_luminance_array[x][y] < ignore_threshold) {
                image_luminance_array[x][y] = 0.0;
                num_ignore += 1;
            }
        }
    }

    auto num_pixels = image_resolution.x * image_resolution.y;

    if (num_ignore == num_pixels) {
        REPORT_FATAL_ERROR();
    }

    if (num_ignore > 0) {
        printf("ImageInfiniteLight::%s(): %d/%d (%.2f%) pixels ignored (ignored ratio: %f)\n",
               __func__, num_ignore, num_pixels, Real(num_ignore) / num_pixels * 100, ignore_ratio);
    }

    image_le_distribution = Distribution2D::create(image_luminance_array, allocator);
}

PBRT_CPU_GPU
SampledSpectrum ImageInfiniteLight::le(const Ray &ray, const SampledWavelengths &lambda) const {
    Vector3f wLight = (render_from_light.apply_inverse(ray.d)).normalize();
    auto uv = EqualAreaSphereToSquare(wLight);
    return ImageLe(uv, lambda);
}

PBRT_CPU_GPU
pbrt::optional<LightLiSample> ImageInfiniteLight::sample_li(const LightSampleContext &ctx,
                                                            const Point2f &u,
                                                            SampledWavelengths &lambda) const {
    const auto [uv, pixel_pdf] = image_le_distribution->sample(u);
    const auto map_pdf = pixel_pdf * image_resolution.x * image_resolution.y;

    Vector3f wLight = EqualAreaSquareToSphere(uv);
    Vector3f wi = render_from_light(wLight);

    // Compute PDF for sampled infinite light direction
    const Real pdf = map_pdf / (4 * compute_pi());

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

PBRT_CPU_GPU
Real ImageInfiniteLight::pdf_li(const LightSampleContext &ctx, const Vector3f &w,
                                bool allow_incomplete_pdf) const {
    Vector3f wLight = render_from_light.apply_inverse(w);
    Point2f uv = EqualAreaSphereToSquare(wLight);

    auto pdf = this->image_le_distribution->get_pdf(uv) * image_resolution.x * image_resolution.y;
    return pdf / (4 * compute_pi());
}

void ImageInfiniteLight::preprocess(const Bounds3f &scene_bounds) {
    scene_bounds.bounding_sphere(&scene_center, &scene_radius);
}

PBRT_CPU_GPU
SampledSpectrum ImageInfiniteLight::ImageLe(Point2f uv, const SampledWavelengths &lambda) const {
    const auto rgb = image_ptr->bilerp(uv, WrapMode::OctahedralSphere).clamp(0, Infinity);

    auto spec = RGBIlluminantSpectrum(rgb, color_space);

    return scale * spec.sample(lambda);
}
