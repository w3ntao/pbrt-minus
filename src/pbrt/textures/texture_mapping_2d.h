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
    TexCoord2D map(const TextureEvalContext &ctx) const {
        const auto PI = compute_pi();
        const auto Inv2Pi = 0.5 / PI;

        Point3f pt = textureFromRender(ctx.p);
        // Compute texture coordinate differentials for cylinder $(u,v)$ mapping
        Real x2y2 = sqr(pt.x) + sqr(pt.y);
        Vector3f dsdp = Vector3f(-pt.y, pt.x, 0) / (2 * PI * x2y2), dtdp = Vector3f(0, 0, 1);

        auto dpdx = textureFromRender(ctx.dpdx);
        auto dpdy = textureFromRender(ctx.dpdy);

        auto dsdx = dsdp.dot(dpdx);
        auto dsdy = dsdp.dot(dpdy);
        auto dtdx = dtdp.dot(dpdx);
        auto dtdy = dtdp.dot(dpdy);

        Point2f st((PI + std::atan2(pt.y, pt.x)) * Inv2Pi, pt.z);
        return TexCoord2D{st, dsdx, dsdy, dtdx, dtdy};
    }

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
    TexCoord2D map(const TextureEvalContext &ctx) const {
        const auto vec = textureFromRender(ctx.p).to_vector3();

        // Initialize partial derivatives of planar mapping $(s,t)$ coordinates
        Vector3f dpdx = textureFromRender(ctx.dpdx);
        Vector3f dpdy = textureFromRender(ctx.dpdy);

        auto dsdx = vs.dot(dpdx);
        auto dsdy = vs.dot(dpdy);
        auto dtdx = vt.dot(dpdx);
        auto dtdy = vt.dot(dpdy);

        Point2f st(ds + vec.dot(vs), dt + vec.dot(vt));
        return TexCoord2D{st, dsdx, dsdy, dtdx, dtdy};
    }

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
    TexCoord2D map(const TextureEvalContext &ctx) const {
        Point3f pt = textureFromRender(ctx.p);
        // Compute $\partial\,s/\partial\,\pt{}$ and $\partial\,t/\partial\,\pt{}$ for
        // spherical mapping

        const auto PI = compute_pi();
        const auto InvPi = 1.0 / PI;
        const auto Inv2Pi = 0.5 / PI;

        Real x2y2 = sqr(pt.x) + sqr(pt.y);
        Real sqrtx2y2 = std::sqrt(x2y2);
        Vector3f dsdp = Vector3f(-pt.y, pt.x, 0) / (2 * PI * x2y2);
        Vector3f dtdp = 1 / (PI * (x2y2 + sqr(pt.z))) *
                        Vector3f(pt.x * pt.z / sqrtx2y2, pt.y * pt.z / sqrtx2y2, -sqrtx2y2);

        // Compute texture coordinate differentials for spherical mapping
        Vector3f dpdx = textureFromRender(ctx.dpdx);
        Vector3f dpdy = textureFromRender(ctx.dpdy);
        Real dsdx = dsdp.dot(dpdx);
        Real dsdy = dsdp.dot(dpdy);
        Real dtdx = dtdp.dot(dpdx);
        Real dtdy = dtdp.dot(dpdy);

        // Return $(s,t)$ texture coordinates and differentials based on spherical mapping
        auto vec = (pt - Point3f(0, 0, 0)).normalize();

        Point2f st(vec.spherical_theta() * InvPi, vec.spherical_phi() * Inv2Pi);

        return TexCoord2D{st, dsdx, dsdy, dtdx, dtdy};
    }

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
    TexCoord2D map(const TextureEvalContext &ctx) const {
        // Compute texture differentials for 2D $(u,v)$ mapping
        Real dsdx = su * ctx.dudx;
        Real dsdy = su * ctx.dudy;
        Real dtdx = sv * ctx.dvdx;
        Real dtdy = sv * ctx.dvdy;

        Point2f st(su * ctx.uv[0] + du, sv * ctx.uv[1] + dv);
        return TexCoord2D{st, dsdx, dsdy, dtdx, dtdy};
    }

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
    Type type;
    const void *ptr;

    void init(const CylindricalMapping *cylindrical_mapping);

    void init(const PlanarMapping *planar_mapping);

    void init(const SphericalMapping *spherical_mapping);

    void init(const UVMapping *uv_mapping);
};
