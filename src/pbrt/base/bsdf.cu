#include "pbrt/base/bsdf.h"
#include "pbrt/base/bxdf.h"
#include "pbrt/bxdfs/diffuse_bxdf.h"

PBRT_GPU void BSDF::init(const Normal3f &ns, const Vector3f &dpdus) {
    shading_frame = Frame::from_xz(dpdus.normalize(), ns.to_vector3());
    bxdf_type = BxDFType::diffuse_bxdf;
    // diffuse_bxdf.init(SampledSpectrum::same_value(1.0));
}

PBRT_GPU
SampledSpectrum BSDF::f(const Vector3f &woRender, const Vector3f &wiRender,
                        const TransportMode mode) const {
    Vector3f wi = RenderToLocal(wiRender);
    Vector3f wo = RenderToLocal(woRender);

    if (wo.z == 0) {
        return SampledSpectrum::same_value(0.0);
    }

    switch (bxdf_type) {
    case (BxDFType::diffuse_bxdf): {
        return diffuse_bxdf.f(wo, wi, mode);
    }
    }

    report_error();
    return SampledSpectrum::same_value(NAN);
}
