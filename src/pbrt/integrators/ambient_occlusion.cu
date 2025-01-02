#include "pbrt/integrators/ambient_occlusion.h"
#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/spectrum_util/global_spectra.h"

const AmbientOcclusionIntegrator *
AmbientOcclusionIntegrator::create(const ParameterDictionary &parameters,
                                   const IntegratorBase *integrator_base,
                                   std::vector<void *> &gpu_dynamic_pointers) {

    auto illuminant_spectrum = parameters.global_spectra->rgb_color_space->illuminant;

    const auto cie_y = parameters.global_spectra->cie_y;
    auto illuminant_scale = 1.0 / illuminant_spectrum->to_photometric(cie_y);

    AmbientOcclusionIntegrator *ambient_occlusion_integrator;
    CHECK_CUDA_ERROR(
        cudaMallocManaged(&ambient_occlusion_integrator, sizeof(AmbientOcclusionIntegrator)));
    gpu_dynamic_pointers.push_back(ambient_occlusion_integrator);

    ambient_occlusion_integrator->init(integrator_base, illuminant_spectrum, illuminant_scale);

    return ambient_occlusion_integrator;
}

PBRT_GPU
SampledSpectrum AmbientOcclusionIntegrator::li(const Ray &ray, SampledWavelengths &lambda,
                                               Sampler *sampler) const {
    const auto shape_intersection = base->intersect(ray, Infinity);

    if (!shape_intersection) {
        return SampledSpectrum(0.0);
    }

    const SurfaceInteraction &isect = shape_intersection->interaction;

    const auto normal = isect.shading.n.to_vector3().face_forward(-ray.d);

    auto u = sampler->get_2d();
    auto local_wi = sample_cosine_hemisphere(u);
    auto pdf = cosine_hemisphere_pdf(std::abs(local_wi.z));

    if (pdf == 0.0) {
        return SampledSpectrum(0.0);
    }

    auto frame = Frame::from_z(normal);
    auto wi = frame.from_local(local_wi);

    // Divide by PI so that fully visible is one.
    auto spawned_ray = isect.spawn_ray(wi);

    if (base->bvh->fast_intersect(spawned_ray, Infinity)) {
        return SampledSpectrum(0.0);
    }

    return illuminant_spectrum->sample(lambda) *
           (illuminant_scale * normal.dot(wi) / (compute_pi() * pdf));
}
