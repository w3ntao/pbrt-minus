#pragma once

#include <pbrt/base/spectrum_texture.h>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/euclidean_space/transform.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>

// TexCoord2D Definition
struct TexCoord2D {
    Point2f st;
    Real dsdx, dsdy, dtdx, dtdy;
};

class CylindricalMapping {
  public:
    static const CylindricalMapping *create(const Transform &texture_from_render,
                                            GPUMemoryAllocator &allocator) {
        auto cylindrical_mapping = allocator.allocate<CylindricalMapping>();
        cylindrical_mapping->textureFromRender = texture_from_render;
        return cylindrical_mapping;
    }

    PBRT_CPU_GPU
    TexCoord2D map(const TextureEvalContext &ctx) const;

  private:
    Transform textureFromRender;
};

class PlanarMapping {
  public:
    static const PlanarMapping *create(const Transform &texture_from_render, const Vector3f &vs,
                                       const Vector3f &vt, Real ds, Real dt,
                                       GPUMemoryAllocator &allocator) {
        auto planar_mapping = allocator.allocate<PlanarMapping>();
        planar_mapping->textureFromRender = texture_from_render;

        planar_mapping->vs = vs;
        planar_mapping->vt = vt;

        planar_mapping->ds = ds;
        planar_mapping->dt = dt;

        return planar_mapping;
    }

    PBRT_CPU_GPU
    TexCoord2D map(const TextureEvalContext &ctx) const;

  private:
    Transform textureFromRender;
    Vector3f vs, vt;
    Real ds, dt;
};

class SphericalMapping {
  public:
    static const SphericalMapping *create(const Transform &texture_from_render,
                                          GPUMemoryAllocator &allocator) {
        auto spherical_mapping = allocator.allocate<SphericalMapping>();
        spherical_mapping->textureFromRender = texture_from_render;

        return spherical_mapping;
    }

    PBRT_CPU_GPU
    TexCoord2D map(const TextureEvalContext &ctx) const;

  private:
    Transform textureFromRender;
};

class UVMapping {
  public:
    static const UVMapping *create(Real su, Real sv, Real du, Real dv,
                                   GPUMemoryAllocator &allocator) {
        auto uv_mapping = allocator.allocate<UVMapping>();
        uv_mapping->init(su, sv, du, dv);

        return uv_mapping;
    }

    PBRT_CPU_GPU
    TexCoord2D map(const TextureEvalContext &ctx) const;

  private:
    Real su, sv, du, dv;

    void init(Real _su, Real _sv, Real _du, Real _dv) {
        su = _su;
        sv = _sv;
        du = _du;
        dv = _dv;
    }
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
    // TODO: TextureMapping2D: rewrite void *ptr with std::variant
    Type type;
    const void *ptr = nullptr;

    void init(const CylindricalMapping *cylindrical_mapping);

    void init(const PlanarMapping *planar_mapping);

    void init(const SphericalMapping *spherical_mapping);

    void init(const UVMapping *uv_mapping);
};
