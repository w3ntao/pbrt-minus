#include "pbrt/base/bxdf.h"

#include "pbrt/bxdfs/diffuse_bxdf.h"
#include "pbrt/bxdfs/dielectric_bxdf.h"
#include "pbrt/bxdfs/coated_diffuse_bxdf.h"

PBRT_GPU
void BxDF::init(const DiffuseBxDF *diffuse_bxdf) {
    type = Type::diffuse_bxdf;
    ptr = diffuse_bxdf;
}

PBRT_GPU
void BxDF::init(const DielectricBxDF *dielectric_bxdf) {
    type = Type::dielectric_bxdf;
    ptr = dielectric_bxdf;
}

PBRT_GPU
void BxDF::init(const CoatedDiffuseBxDF *coated_diffuse_bxdf) {
    type = Type::coated_diffuse_bxdf;
    ptr = coated_diffuse_bxdf;
}

PBRT_CPU_GPU
BxDFFlags BxDF::flags() const {
    switch (type) {
    case (Type::diffuse_bxdf): {
        return ((DiffuseBxDF *)ptr)->flags();
    }

    case (Type::dielectric_bxdf): {
        return ((DielectricBxDF *)ptr)->flags();
    }

    case (Type::coated_diffuse_bxdf): {
        return ((CoatedDiffuseBxDF *)ptr)->flags();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_GPU
SampledSpectrum BxDF::f(Vector3f wo, Vector3f wi, TransportMode mode) const {
    switch (type) {
    case (Type::diffuse_bxdf): {
        return ((DiffuseBxDF *)ptr)->f(wo, wi, mode);
    }

    case (Type::dielectric_bxdf): {
        return ((DielectricBxDF *)ptr)->f(wo, wi, mode);
    }

    case (Type::coated_diffuse_bxdf): {
        return ((CoatedDiffuseBxDF *)ptr)->f(wo, wi, mode);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_GPU
cuda::std::optional<BSDFSample> BxDF::sample_f(Vector3f wo, FloatType uc, Point2f u,
                                               TransportMode mode,
                                               BxDFReflTransFlags sampleFlags) const {
    switch (type) {
    case (Type::diffuse_bxdf): {
        return ((DiffuseBxDF *)ptr)->sample_f(wo, uc, u, mode, sampleFlags);
    }

    case (Type::dielectric_bxdf): {
        return ((DielectricBxDF *)ptr)->sample_f(wo, uc, u, mode, sampleFlags);
    }

    case (Type::coated_diffuse_bxdf): {
        return ((CoatedDiffuseBxDF *)ptr)->sample_f(wo, uc, u, mode, sampleFlags);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_GPU
FloatType BxDF::pdf(Vector3f wo, Vector3f wi, TransportMode mode,
                    BxDFReflTransFlags sampleFlags) const {
    switch (type) {
    case (Type::diffuse_bxdf): {
        return ((DiffuseBxDF *)ptr)->pdf(wo, wi, mode, sampleFlags);
    }

    case (Type::dielectric_bxdf): {
        return ((DielectricBxDF *)ptr)->pdf(wo, wi, mode, sampleFlags);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
