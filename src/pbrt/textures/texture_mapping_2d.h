#pragma once

#include <cuda/std/variant>
#include <pbrt/base/spectrum_texture.h>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/euclidean_space/transform.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>

// TexCoord2D Definition
struct TexCoord2D {
    Point2f st{NAN, NAN};
    Real dsdx = NAN;
    Real dsdy = NAN;
    Real dtdx = NAN;
    Real dtdy = NAN;
};

class CylindricalMapping {
  public:
    CylindricalMapping(const Transform &texture_from_render)
        : textureFromRender(texture_from_render) {}

    PBRT_CPU_GPU
    TexCoord2D map(const TextureEvalContext &ctx) const;

  private:
    Transform textureFromRender;
};

class PlanarMapping {
  public:
    PlanarMapping(const Transform &texture_from_render, const Vector3f &_vs, const Vector3f &_vt,
                  Real _ds, Real _dt)
        : textureFromRender(texture_from_render), vs(_vs), vt(_vt), ds(_ds), dt(_dt) {}

    PBRT_CPU_GPU
    TexCoord2D map(const TextureEvalContext &ctx) const;

  private:
    Transform textureFromRender;
    Vector3f vs, vt;
    Real ds = NAN;
    Real dt = NAN;
};

class SphericalMapping {
  public:
    SphericalMapping(const Transform &texture_from_render)
        : textureFromRender(texture_from_render) {}

    PBRT_CPU_GPU
    TexCoord2D map(const TextureEvalContext &ctx) const;

  private:
    Transform textureFromRender;
};

class UVMapping {
  public:
    UVMapping(Real _su, Real _sv, Real _du, Real _dv) : su(_su), sv(_sv), du(_du), dv(_dv) {}

    PBRT_CPU_GPU
    TexCoord2D map(const TextureEvalContext &ctx) const;

  private:
    Real su = NAN;
    Real sv = NAN;
    Real du = NAN;
    Real dv = NAN;
};

struct TextureMapping2D {
    enum class Type {
        cylindrical,
        planar,
        spherical,
        uv,
    };

    static const TextureMapping2D *create(const Transform &render_from_texture,
                                          const ParameterDictionary &parameters,
                                          GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    TexCoord2D map(const TextureEvalContext &ctx) const;

  private:
    Type type;
    const void *ptr = nullptr;

    void init(const CylindricalMapping *cylindrical_mapping);

    void init(const PlanarMapping *planar_mapping);

    void init(const SphericalMapping *spherical_mapping);

    void init(const UVMapping *uv_mapping);
};
