#pragma once

#include <pbrt/base/spectrum_texture.h>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/euclidean_space/transform.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/gpu/macro.h>
#include <pbrt/scene/parameter_dictionary.h>

// TexCoord2D Definition
struct TexCoord2D {
    Point2f st;
    FloatType dsdx, dsdy, dtdx, dtdy;
};

// UVMapping Definition
class UVMapping {
  public:
    static const UVMapping *create(const ParameterDictionary &parameters,
                                   GPUMemoryAllocator &allocator) {
        auto uv_mapping = allocator.allocate<UVMapping>();
        uv_mapping->init(parameters);

        return uv_mapping;
    }

    static const UVMapping *create(FloatType su, FloatType sv, FloatType du, FloatType dv,
                                   GPUMemoryAllocator &allocator) {

        auto uv_mapping = allocator.allocate<UVMapping>();
        uv_mapping->init(su, sv, du, dv);

        return uv_mapping;
    }

    PBRT_CPU_GPU
    TexCoord2D map(const TextureEvalContext &ctx) const {
        // Compute texture differentials for 2D $(u,v)$ mapping
        FloatType dsdx = su * ctx.dudx;
        FloatType dsdy = su * ctx.dudy;
        FloatType dtdx = sv * ctx.dvdx;
        FloatType dtdy = sv * ctx.dvdy;

        Point2f st(su * ctx.uv[0] + du, sv * ctx.uv[1] + dv);
        return TexCoord2D{st, dsdx, dsdy, dtdx, dtdy};
    }

  private:
    FloatType su, sv, du, dv;

    void init(const ParameterDictionary &parameters) {
        su = parameters.get_float("uscale", 1.0);
        sv = parameters.get_float("vscale", 1.0);
        du = parameters.get_float("udelta", 0.0);
        dv = parameters.get_float("vdelta", 0.0);
    }

    void init(FloatType _su, FloatType _sv, FloatType _du, FloatType _dv) {
        su = _su;
        sv = _sv;
        du = _du;
        dv = _dv;
    }
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
    TexCoord2D map(const TextureEvalContext &ctx) const {
        Point3f pt = textureFromRender(ctx.p);
        // Compute $\partial\,s/\partial\,\pt{}$ and $\partial\,t/\partial\,\pt{}$ for
        // spherical mapping

        const auto PI = compute_pi();
        const auto InvPi = 1.0 / PI;
        const auto Inv2Pi = 0.5 / PI;

        FloatType x2y2 = sqr(pt.x) + sqr(pt.y);
        FloatType sqrtx2y2 = std::sqrt(x2y2);
        Vector3f dsdp = Vector3f(-pt.y, pt.x, 0) / (2 * PI * x2y2);
        Vector3f dtdp = 1 / (PI * (x2y2 + sqr(pt.z))) *
                        Vector3f(pt.x * pt.z / sqrtx2y2, pt.y * pt.z / sqrtx2y2, -sqrtx2y2);

        // Compute texture coordinate differentials for spherical mapping
        Vector3f dpdx = textureFromRender(ctx.dpdx);
        Vector3f dpdy = textureFromRender(ctx.dpdy);
        FloatType dsdx = dsdp.dot(dpdx);
        FloatType dsdy = dsdp.dot(dpdy);
        FloatType dtdx = dtdp.dot(dpdx);
        FloatType dtdy = dtdp.dot(dpdy);

        // Return $(s,t)$ texture coordinates and differentials based on spherical mapping
        auto vec = (pt - Point3f(0, 0, 0)).normalize();

        Point2f st(vec.spherical_theta() * InvPi, vec.spherical_phi() * Inv2Pi);

        return TexCoord2D{st, dsdx, dsdy, dtdx, dtdy};
    }

  private:
    Transform textureFromRender;
};

struct TextureMapping2D {
    enum class Type {
        uv,
        spherical,
    };

    static const TextureMapping2D *create(const Transform &render_from_texture,
                                          const ParameterDictionary &parameters,
                                          GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    TexCoord2D map(const TextureEvalContext &ctx) const;

  private:
    Type type;
    const void *ptr;

    void init(const UVMapping *uv_mapping);

    void init(const SphericalMapping *spherical_mapping);
};
