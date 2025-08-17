#include <pbrt/base/spectrum.h>
#include <pbrt/euclidean_space/transform.h>
#include <pbrt/lights/distant_light.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/spectrum_util/rgb_color_space.h>

#include <pbrt/gpu/gpu_memory_allocator.h>

DistantLight *DistantLight::create(const Transform &renderFromLight,
                                   const ParameterDictionary &parameters,
                                   GPUMemoryAllocator &allocator) {
    auto Lemit = parameters.get_spectrum("L", SpectrumType::Illuminant, allocator);
    if (Lemit == nullptr) {
        Lemit = parameters.global_spectra->rgb_color_space->illuminant;
    }

    auto scale = parameters.get_float("scale", 1.0);

    Point3f from = parameters.get_point3("from", Point3f(0, 0, 0));
    Point3f to = parameters.get_point3("to", Point3f(0, 0, 1));

    Vector3f w = (from - to).normalize();
    Vector3f v1, v2;
    w.coordinate_system(&v1, &v2);

    Real m[4][4] = {v1.x, v2.x, w.x, 0, v1.y, v2.y, w.y, 0, v1.z, v2.z, w.z, 0, 0, 0, 0, 1};
    auto t = Transform(m);

    const Transform final_render_from_light = renderFromLight * t;

    scale /= Lemit->to_photometric(parameters.global_spectra->cie_y);

    const auto E_v = parameters.get_float("illuminance", -1);
    if (E_v > 0) {
        scale *= E_v;
    }

    return allocator.create<DistantLight>(final_render_from_light, Lemit, scale);
}

PBRT_CPU_GPU
pbrt::optional<LightLiSample> DistantLight::sample_li(const LightSampleContext &ctx,
                                                      const Point2f &u,
                                                      SampledWavelengths &lambda) const {
    Vector3f wi = render_from_light(Vector3f(0, 0, 1)).normalize();
    Point3f pOutside = ctx.p() + wi * (2 * scene_radius);

    return LightLiSample(scale * Lemit->sample(lambda), wi, 1, Interaction(pOutside));
}

PBRT_CPU_GPU
SampledSpectrum DistantLight::phi(const SampledWavelengths &lambda) const {
    return scale * Lemit->sample(lambda) * sqr(scene_radius);
}
