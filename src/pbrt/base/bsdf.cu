#include "pbrt/base/bsdf.h"

#include "pbrt/base/bxdf.h"
#include "pbrt/bxdfs/diffuse_bxdf.h"

PBRT_GPU
void BSDF::init_frame(const Normal3f &ns, const Vector3f &dpdus) {
    shading_frame = Frame::from_xz(dpdus.normalize(), ns.to_vector3());
}

PBRT_GPU
void BSDF::init_bxdf(const DiffuseBxDF *diffuse_bxdf) {
    bxdf.init(diffuse_bxdf);
}

PBRT_GPU
SampledSpectrum BSDF::f(const Vector3f &woRender, const Vector3f &wiRender,
                        const TransportMode mode) const {
    Vector3f wi = RenderToLocal(wiRender);
    Vector3f wo = RenderToLocal(woRender);

    if (wo.z == 0) {
        return SampledSpectrum::same_value(0.0);
    }

    return bxdf.f(wo, wi, mode);
}
