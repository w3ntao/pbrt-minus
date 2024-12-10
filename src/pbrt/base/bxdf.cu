#include "pbrt/base/bxdf.h"
#include "pbrt/bxdfs/coated_conductor_bxdf.h"
#include "pbrt/bxdfs/coated_diffuse_bxdf.h"
#include "pbrt/bxdfs/conductor_bxdf.h"
#include "pbrt/bxdfs/dielectric_bxdf.h"
#include "pbrt/bxdfs/diffuse_bxdf.h"

PBRT_GPU
void BxDF::init(CoatedConductorBxDF *coated_conductor_bxdf) {
    type = Type::coated_conductor;
    ptr = coated_conductor_bxdf;
}

PBRT_GPU
void BxDF::init(CoatedDiffuseBxDF *coated_diffuse_bxdf) {
    type = Type::coated_diffuse;
    ptr = coated_diffuse_bxdf;
}

PBRT_GPU
void BxDF::init(ConductorBxDF *conductor_bxdf) {
    type = Type::conductor;
    ptr = conductor_bxdf;
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
    case Type::coated_conductor: {
        return static_cast<CoatedConductorBxDF *>(ptr)->flags();
    }

    case Type::coated_diffuse: {
        return static_cast<CoatedDiffuseBxDF *>(ptr)->flags();
    }

    case Type::conductor: {
        return static_cast<ConductorBxDF *>(ptr)->flags();
    }

    case Type::dielectric: {
        return static_cast<DielectricBxDF *>(ptr)->flags();
    }

    case Type::diffuse: {
        return static_cast<DiffuseBxDF *>(ptr)->flags();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_GPU
void BxDF::regularize() {
    switch (type) {
    case Type::coated_conductor: {
        static_cast<CoatedConductorBxDF *>(ptr)->regularize();
        return;
    }

    case Type::coated_diffuse: {
        static_cast<CoatedDiffuseBxDF *>(ptr)->regularize();
        return;
    }

    case Type::conductor: {
        return static_cast<ConductorBxDF *>(ptr)->regularize();
    }

    case Type::dielectric: {
        return static_cast<DielectricBxDF *>(ptr)->regularize();
    }

    case Type::diffuse: {
        return static_cast<DiffuseBxDF *>(ptr)->regularize();
    }
    }
    REPORT_FATAL_ERROR();
}

PBRT_GPU
SampledSpectrum BxDF::f(Vector3f wo, Vector3f wi, TransportMode mode) const {
    switch (type) {
    case Type::coated_conductor: {
        return static_cast<CoatedConductorBxDF *>(ptr)->f(wo, wi, mode);
    }

    case Type::coated_diffuse: {
        return static_cast<CoatedDiffuseBxDF *>(ptr)->f(wo, wi, mode);
    }

    case Type::conductor: {
        return static_cast<ConductorBxDF *>(ptr)->f(wo, wi, mode);
    }

    case Type::dielectric: {
        return static_cast<DielectricBxDF *>(ptr)->f(wo, wi, mode);
    }

    case Type::diffuse: {
        return static_cast<DiffuseBxDF *>(ptr)->f(wo, wi, mode);
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
    case Type::coated_conductor: {
        return static_cast<CoatedConductorBxDF *>(ptr)->sample_f(wo, uc, u, mode, sampleFlags);
    }

    case Type::coated_diffuse: {
        return static_cast<CoatedDiffuseBxDF *>(ptr)->sample_f(wo, uc, u, mode, sampleFlags);
    }

    case Type::conductor: {
        return static_cast<ConductorBxDF *>(ptr)->sample_f(wo, uc, u, mode, sampleFlags);
    }

    case Type::dielectric: {
        return static_cast<DielectricBxDF *>(ptr)->sample_f(wo, uc, u, mode, sampleFlags);
    }

    case Type::diffuse: {
        return static_cast<DiffuseBxDF *>(ptr)->sample_f(wo, uc, u, mode, sampleFlags);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_GPU
FloatType BxDF::pdf(Vector3f wo, Vector3f wi, TransportMode mode,
                    BxDFReflTransFlags sampleFlags) const {
    switch (type) {
    case Type::coated_conductor: {
        return static_cast<CoatedConductorBxDF *>(ptr)->pdf(wo, wi, mode, sampleFlags);
    }

    case Type::coated_diffuse: {
        return static_cast<CoatedDiffuseBxDF *>(ptr)->pdf(wo, wi, mode, sampleFlags);
    }

    case Type::conductor: {
        return static_cast<ConductorBxDF *>(ptr)->pdf(wo, wi, mode, sampleFlags);
    }

    case Type::diffuse: {
        return static_cast<DiffuseBxDF *>(ptr)->pdf(wo, wi, mode, sampleFlags);
    }

    case Type::dielectric: {
        return static_cast<DielectricBxDF *>(ptr)->pdf(wo, wi, mode, sampleFlags);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
