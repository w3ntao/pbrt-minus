#include "pbrt/base/bxdf.h"

#include "pbrt/bxdfs/coated_diffuse_bxdf.h"
#include "pbrt/bxdfs/conductor_bxdf.h"
#include "pbrt/bxdfs/dielectric_bxdf.h"
#include "pbrt/bxdfs/diffuse_bxdf.h"

PBRT_GPU
void BxDF::init(ConductorBxDF *conductor_bxdf) {
    type = Type::conductor;
    ptr = conductor_bxdf;
}

PBRT_GPU
void BxDF::init(CoatedDiffuseBxDF *coated_diffuse_bxdf) {
    type = Type::coated_diffuse;
    ptr = coated_diffuse_bxdf;
}

PBRT_GPU
void BxDF::init(DielectricBxDF *dielectric_bxdf) {
    type = Type::dielectric;
    ptr = dielectric_bxdf;
}

PBRT_GPU
void BxDF::init(DiffuseBxDF *diffuse_bxdf) {
    type = Type::diffuse;
    ptr = diffuse_bxdf;
}

PBRT_CPU_GPU
BxDFFlags BxDF::flags() const {
    switch (type) {
    case (Type::coated_diffuse): {
        return ((CoatedDiffuseBxDF *)ptr)->flags();
    }

    case (Type::conductor): {
        return ((ConductorBxDF *)ptr)->flags();
    }

    case (Type::dielectric): {
        return ((DielectricBxDF *)ptr)->flags();
    }

    case (Type::diffuse): {
        return ((DiffuseBxDF *)ptr)->flags();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_GPU
void BxDF::regularize() {
    switch (type) {
    case (Type::coated_diffuse): {
        ((CoatedDiffuseBxDF *)ptr)->regularize();
        return;
    }

    case (Type::conductor): {
        return ((ConductorBxDF *)ptr)->regularize();
    }

    case (Type::dielectric): {
        return ((DielectricBxDF *)ptr)->regularize();
    }

    case (Type::diffuse): {
        return ((DiffuseBxDF *)ptr)->regularize();
    }
    }
    REPORT_FATAL_ERROR();
}

PBRT_GPU
SampledSpectrum BxDF::f(Vector3f wo, Vector3f wi, TransportMode mode) const {
    switch (type) {
    case (Type::coated_diffuse): {
        return ((CoatedDiffuseBxDF *)ptr)->f(wo, wi, mode);
    }

    case (Type::conductor): {
        return ((ConductorBxDF *)ptr)->f(wo, wi, mode);
    }

    case (Type::dielectric): {
        return ((DielectricBxDF *)ptr)->f(wo, wi, mode);
    }

    case (Type::diffuse): {
        return ((DiffuseBxDF *)ptr)->f(wo, wi, mode);
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
    case (Type::coated_diffuse): {
        return ((CoatedDiffuseBxDF *)ptr)->sample_f(wo, uc, u, mode, sampleFlags);
    }

    case (Type::conductor): {
        return ((ConductorBxDF *)ptr)->sample_f(wo, uc, u, mode, sampleFlags);
    }

    case (Type::dielectric): {
        return ((DielectricBxDF *)ptr)->sample_f(wo, uc, u, mode, sampleFlags);
    }

    case (Type::diffuse): {
        return ((DiffuseBxDF *)ptr)->sample_f(wo, uc, u, mode, sampleFlags);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_GPU
FloatType BxDF::pdf(Vector3f wo, Vector3f wi, TransportMode mode,
                    BxDFReflTransFlags sampleFlags) const {
    switch (type) {
    case (Type::coated_diffuse): {
        return ((CoatedDiffuseBxDF *)ptr)->pdf(wo, wi, mode, sampleFlags);
    }

    case (Type::conductor): {
        return ((ConductorBxDF *)ptr)->pdf(wo, wi, mode, sampleFlags);
    }

    case (Type::diffuse): {
        return ((DiffuseBxDF *)ptr)->pdf(wo, wi, mode, sampleFlags);
    }

    case (Type::dielectric): {
        return ((DielectricBxDF *)ptr)->pdf(wo, wi, mode, sampleFlags);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
