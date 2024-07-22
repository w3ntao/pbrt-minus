#include "pbrt/lights/distant_light.h"

#include "pbrt/base/spectrum.h"
#include "pbrt/euclidean_space/transform.h"
#include "pbrt/spectrum_util/global_spectra.h"
#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/spectra/rgb_illuminant_spectrum.h"

DistantLight *DistantLight::create(const Transform &renderFromLight,
                                   const ParameterDictionary &parameters,
                                   std::vector<void *> &gpu_dynamic_pointers) {

    DistantLight *distant_light;
    CHECK_CUDA_ERROR(cudaMallocManaged(&distant_light, sizeof(DistantLight)));
    gpu_dynamic_pointers.push_back(distant_light);

    auto lemit = parameters.get_spectrum("L", SpectrumType::Illuminant, gpu_dynamic_pointers);
    if (lemit == nullptr) {
        lemit = parameters.global_spectra->rgb_color_space->illuminant;
    }

    auto sc = parameters.get_float("scale", 1.0);

    Point3f from = parameters.get_point3("from", Point3f(0, 0, 0));
    Point3f to = parameters.get_point3("to", Point3f(0, 0, 1));

    Vector3f w = (from - to).normalize();
    Vector3f v1, v2;
    w.coordinate_system(&v1, &v2);

    FloatType m[4][4] = {v1.x, v2.x, w.x, 0, v1.y, v2.y, w.y, 0, v1.z, v2.z, w.z, 0, 0, 0, 0, 1};
    auto t = Transform(m);

    Transform finalRenderFromLight = renderFromLight * t;

    sc /= lemit->to_photometric(parameters.global_spectra->cie_y);

    auto E_v = parameters.get_float("illuminance", -1);
    if (E_v > 0) {
        sc *= E_v;
    }

    distant_light->light_type = LightType::delta_direction;
    distant_light->render_from_light = finalRenderFromLight;

    distant_light->scale = sc;
    distant_light->l_emit = lemit;

    distant_light->scene_radius = NAN;
    distant_light->scene_center = Point3f(NAN, NAN, NAN);

    return distant_light;
}

PBRT_GPU
cuda::std::optional<LightLiSample> DistantLight::sample_li(const LightSampleContext &ctx,
                                                           const Point2f &u,
                                                           SampledWavelengths &lambda) const {
    Vector3f wi = render_from_light(Vector3f(0, 0, 1)).normalize();
    Point3f pOutside = ctx.p() + wi * (2 * scene_radius);

    return LightLiSample(scale * l_emit->sample(lambda), wi, 1, Interaction(pOutside));
}
