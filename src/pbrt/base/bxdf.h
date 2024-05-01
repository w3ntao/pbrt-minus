#pragma once

#include <cuda/std/optional>

#include "pbrt/euclidean_space/vector3.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"
#include "pbrt/util/macro.h"

class DiffuseBxDF;

enum BxDFFlags {
    Unset = 0,
    Reflection = 1 << 0,
    Transmission = 1 << 1,
    Diffuse = 1 << 2,
    Glossy = 1 << 3,
    Specular = 1 << 4,
    // Composite _BxDFFlags_ definitions
    DiffuseReflection = Diffuse | Reflection,
    DiffuseTransmission = Diffuse | Transmission,
    GlossyReflection = Glossy | Reflection,
    GlossyTransmission = Glossy | Transmission,
    SpecularReflection = Specular | Reflection,
    SpecularTransmission = Specular | Transmission,
    All = Diffuse | Glossy | Specular | Reflection | Transmission
};

namespace {
// BxDFFlags Inline Functions
PBRT_CPU_GPU inline bool _is_reflective(BxDFFlags f) {
    return f & BxDFFlags::Reflection;
}
PBRT_CPU_GPU inline bool _is_transmissive(BxDFFlags f) {
    return f & BxDFFlags::Transmission;
}
PBRT_CPU_GPU inline bool _is_diffuse(BxDFFlags f) {
    return f & BxDFFlags::Diffuse;
}
PBRT_CPU_GPU inline bool _is_glossy(BxDFFlags f) {
    return f & BxDFFlags::Glossy;
}
PBRT_CPU_GPU inline bool _is_specular(BxDFFlags f) {
    return f & BxDFFlags::Specular;
}
PBRT_CPU_GPU inline bool _is_non_specular(BxDFFlags f) {
    return f & (BxDFFlags::Diffuse | BxDFFlags::Glossy);
}

} // namespace

enum class BxDFReflTransFlags {
    Unset = 0,
    Reflection = 1 << 0,
    Transmission = 1 << 1,
    All = Reflection | Transmission
};

PBRT_CPU_GPU
inline BxDFReflTransFlags operator|(BxDFReflTransFlags a, BxDFReflTransFlags b) {
    return BxDFReflTransFlags((int)a | (int)b);
}

PBRT_CPU_GPU
inline int operator&(BxDFReflTransFlags a, BxDFReflTransFlags b) {
    return ((int)a & (int)b);
}

PBRT_CPU_GPU
inline BxDFReflTransFlags &operator|=(BxDFReflTransFlags &a, BxDFReflTransFlags b) {
    (int &)a |= int(b);
    return a;
}

// TransportMode Definition
enum class TransportMode {
    Radiance,
    Importance,
};

PBRT_CPU_GPU
inline TransportMode operator!(TransportMode mode) {
    return (mode == TransportMode::Radiance) ? TransportMode::Importance : TransportMode::Radiance;
}

// BSDFSample Definition
struct BSDFSample {
    // BSDFSample Public Methods

    PBRT_CPU_GPU
    BSDFSample() : pdf(0), eta(1), pdfIsProportional(false) {}

    PBRT_CPU_GPU
    BSDFSample(SampledSpectrum f, Vector3f wi, FloatType pdf, BxDFFlags flags, FloatType eta = 1,
               bool pdfIsProportional = false)
        : f(f), wi(wi), pdf(pdf), flags(flags), eta(eta), pdfIsProportional(pdfIsProportional) {}

    PBRT_CPU_GPU
    bool IsReflection() const {
        return ::_is_reflective(flags);
    }
    PBRT_CPU_GPU
    bool IsTransmission() const {
        return ::_is_transmissive(flags);
    }
    PBRT_CPU_GPU
    bool IsDiffuse() const {
        return ::_is_diffuse(flags);
    }
    PBRT_CPU_GPU
    bool IsGlossy() const {
        return ::_is_glossy(flags);
    }
    PBRT_CPU_GPU
    bool IsSpecular() const {
        return ::_is_specular(flags);
    }

    SampledSpectrum f;
    Vector3f wi;
    FloatType pdf = 0;
    BxDFFlags flags;
    FloatType eta = 1;
    bool pdfIsProportional = false;
};

class BxDF {
  public:
    enum class Type {
        null,
        diffuse_bxdf,
    };

    PBRT_GPU
    BxDF() : bxdf_type(Type::null), bxdf_ptr(nullptr) {}

    PBRT_GPU
    void init(const DiffuseBxDF *diffuse_bxdf);

    PBRT_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const;

    PBRT_GPU
    cuda::std::optional<BSDFSample>
    sample_f(Vector3f wo, FloatType uc, Point2f u, TransportMode mode = TransportMode::Radiance,
             BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    PBRT_GPU
    FloatType pdf(Vector3f wo, Vector3f wi, TransportMode mode,
                  BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    PBRT_GPU
    bool has_type_null() const {
        return bxdf_type == Type::null;
    }

  private:
    Type bxdf_type;
    const void *bxdf_ptr;
};
