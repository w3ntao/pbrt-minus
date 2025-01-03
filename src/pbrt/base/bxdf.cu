#include "pbrt/base/bxdf.h"

PBRT_CPU_GPU
void BxDF::init(const CoatedConductorBxDF &_coated_conductor_bxdf) {
    type = Type::coated_conductor;
    coated_conductor_bxdf = _coated_conductor_bxdf;
}

PBRT_CPU_GPU
void BxDF::init(const CoatedDiffuseBxDF &_coated_diffuse_bxdf) {
    type = Type::coated_diffuse;
    coated_diffuse_bxdf = _coated_diffuse_bxdf;
}

PBRT_CPU_GPU
void BxDF::init(const ConductorBxDF &_conductor_bxdf) {
    type = Type::conductor;
    conductor_bxdf = _conductor_bxdf;
}

PBRT_CPU_GPU
void BxDF::init(const DielectricBxDF &_dielectric_bxdf) {
    type = Type::dielectric;
    dielectric_bxdf = _dielectric_bxdf;
}

PBRT_CPU_GPU
void BxDF::init(const DiffuseBxDF &_diffuse_bxdf) {
    type = Type::diffuse;
    diffuse_bxdf = _diffuse_bxdf;
}

PBRT_CPU_GPU
BxDFFlags BxDF::flags() const {
    switch (type) {
    case Type::coated_conductor: {
        return coated_conductor_bxdf.flags();
    }

    case Type::coated_diffuse: {
        return coated_diffuse_bxdf.flags();
    }

    case Type::conductor: {
        return conductor_bxdf.flags();
    }

    case Type::dielectric: {
        return dielectric_bxdf.flags();
    }

    case Type::diffuse: {
        return diffuse_bxdf.flags();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_GPU
void BxDF::regularize() {
    switch (type) {
    case Type::coated_conductor: {
        coated_conductor_bxdf.regularize();
        return;
    }

    case Type::coated_diffuse: {
        coated_diffuse_bxdf.regularize();
        return;
    }

    case Type::conductor: {
        conductor_bxdf.regularize();
        return;
    }

    case Type::dielectric: {
        dielectric_bxdf.regularize();
        return;
    }

    case Type::diffuse: {
        diffuse_bxdf.regularize();
        return;
    }
    }
    REPORT_FATAL_ERROR();
}

PBRT_GPU
SampledSpectrum BxDF::f(Vector3f wo, Vector3f wi, TransportMode mode) const {
    switch (type) {
    case Type::coated_conductor: {
        return coated_conductor_bxdf.f(wo, wi, mode);
    }

    case Type::coated_diffuse: {
        return coated_diffuse_bxdf.f(wo, wi, mode);
    }

    case Type::conductor: {
        return conductor_bxdf.f(wo, wi, mode);
    }

    case Type::dielectric: {
        return dielectric_bxdf.f(wo, wi, mode);
    }

    case Type::diffuse: {
        return diffuse_bxdf.f(wo, wi, mode);
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
        return coated_conductor_bxdf.sample_f(wo, uc, u, mode, sampleFlags);
    }

    case Type::coated_diffuse: {
        return coated_diffuse_bxdf.sample_f(wo, uc, u, mode, sampleFlags);
    }

    case Type::conductor: {
        return conductor_bxdf.sample_f(wo, uc, u, mode, sampleFlags);
    }

    case Type::dielectric: {
        return dielectric_bxdf.sample_f(wo, uc, u, mode, sampleFlags);
    }

    case Type::diffuse: {
        return diffuse_bxdf.sample_f(wo, uc, u, mode, sampleFlags);
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
        return coated_conductor_bxdf.pdf(wo, wi, mode, sampleFlags);
    }

    case Type::coated_diffuse: {
        return coated_diffuse_bxdf.pdf(wo, wi, mode, sampleFlags);
    }

    case Type::conductor: {
        return conductor_bxdf.pdf(wo, wi, mode, sampleFlags);
    }

    case Type::diffuse: {
        return diffuse_bxdf.pdf(wo, wi, mode, sampleFlags);
    }

    case Type::dielectric: {
        return dielectric_bxdf.pdf(wo, wi, mode, sampleFlags);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
