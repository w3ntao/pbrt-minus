#pragma once

#include "pbrt/base/spectrum.h"
#include "pbrt/util/macro.h"

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
    BSDFSample() : pdf(0), eta(1), pdf_is_proportional(false), wi(Vector3f(NAN, NAN, NAN)) {}

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
    FloatType pdf;
    BxDFFlags flags;
    FloatType eta;
    bool pdf_is_proportional;
};
