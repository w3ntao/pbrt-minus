#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/vector3.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"
#include "pbrt/util/macro.h"
#include <cuda/std/optional>

class CoatedConductorBxDF;
class CoatedDiffuseBxDF;
class ConductorBxDF;
class DielectricBxDF;
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
    All = Reflection | Transmission,
};

PBRT_CPU_GPU
inline BxDFFlags operator|(BxDFFlags a, BxDFFlags b) {
    return BxDFFlags((int)a | (int)b);
}

PBRT_CPU_GPU
inline int operator&(BxDFFlags a, BxDFFlags b) {
    return ((int)a & (int)b);
}

PBRT_CPU_GPU
inline int operator&(BxDFFlags a, BxDFReflTransFlags b) {
    return ((int)a & (int)b);
}

PBRT_CPU_GPU
inline BxDFFlags &operator|=(BxDFFlags &a, BxDFFlags b) {
    (int &)a |= int(b);
    return a;
}

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
    BSDFSample() : pdf(0), eta(1), pdf_is_proportional(false) {}

    PBRT_CPU_GPU
    BSDFSample(SampledSpectrum f, Vector3f wi, FloatType pdf, BxDFFlags flags, FloatType eta = 1,
               bool _pdf_is_proportional = false)
        : f(f), wi(wi), pdf(pdf), flags(flags), eta(eta),
          pdf_is_proportional(_pdf_is_proportional) {}

    PBRT_CPU_GPU
    bool is_reflection() const {
        return ::_is_reflective(flags);
    }
    PBRT_CPU_GPU
    bool is_transmission() const {
        return ::_is_transmissive(flags);
    }
    PBRT_CPU_GPU
    bool is_diffuse() const {
        return ::_is_diffuse(flags);
    }
    PBRT_CPU_GPU
    bool is_glossy() const {
        return ::_is_glossy(flags);
    }
    PBRT_CPU_GPU
    bool is_specular() const {
        return ::_is_specular(flags);
    }

    SampledSpectrum f;
    Vector3f wi;
    FloatType pdf = 0;
    BxDFFlags flags;
    FloatType eta = 1;
    bool pdf_is_proportional = false;
};

class BxDF {
  public:
    enum class Type {
        null,
        coated_conductor,
        coated_diffuse,
        conductor,
        dielectric,
        diffuse,
    };

    PBRT_GPU
    BxDF() : type(Type::null), ptr(nullptr) {}

    PBRT_GPU
    void init(CoatedConductorBxDF *coated_conductor_bxdf);

    PBRT_GPU
    void init(CoatedDiffuseBxDF *coated_diffuse_bxdf);

    PBRT_GPU
    void init(ConductorBxDF *conductor_bxdf);

    PBRT_GPU
    void init(DielectricBxDF *dielectric_bxdf);

    PBRT_GPU
    void init(DiffuseBxDF *diffuse_bxdf);

    PBRT_CPU_GPU
    BxDFFlags flags() const;

    PBRT_GPU
    void regularize();

    PBRT_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const;

    PBRT_GPU
    cuda::std::optional<BSDFSample>
    sample_f(Vector3f wo, FloatType uc, Point2f u, TransportMode mode = TransportMode::Radiance,
             BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    PBRT_GPU
    FloatType pdf(Vector3f wo, Vector3f wi, TransportMode mode,
                  BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    PBRT_CPU_GPU
    bool has_type_null() const {
        return type == Type::null;
    }

  private:
    Type type;
    void *ptr;
};
