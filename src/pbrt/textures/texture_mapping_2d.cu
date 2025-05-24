#include <pbrt/textures/texture_eval_context.h>
#include <pbrt/textures/texture_mapping_2d.h>

PBRT_CPU_GPU
TexCoord2D CylindricalMapping::map(const TextureEvalContext &ctx) const {
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

PBRT_CPU_GPU
TexCoord2D PlanarMapping::map(const TextureEvalContext &ctx) const {
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

PBRT_CPU_GPU
TexCoord2D SphericalMapping::map(const TextureEvalContext &ctx) const {
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

PBRT_CPU_GPU
TexCoord2D UVMapping::map(const TextureEvalContext &ctx) const {
    // Compute texture differentials for 2D $(u,v)$ mapping
    Real dsdx = su * ctx.dudx;
    Real dsdy = su * ctx.dudy;
    Real dtdx = sv * ctx.dvdx;
    Real dtdy = sv * ctx.dvdy;

    Point2f st(su * ctx.uv[0] + du, sv * ctx.uv[1] + dv);
    return TexCoord2D{st, dsdx, dsdy, dtdx, dtdy};
}

const TextureMapping2D *TextureMapping2D::create(const Transform &render_from_texture,
                                                 const ParameterDictionary &parameters,
                                                 GPUMemoryAllocator &allocator) {
    auto texture_mapping = allocator.allocate<TextureMapping2D>();

    const auto type = parameters.get_one_string("mapping", "uv");
    if (type == "cylindrical") {
        *texture_mapping = CylindricalMapping(render_from_texture.inverse());
        return texture_mapping;
    }

    if (type == "planar") {
        const auto v1 = parameters.get_vector3f("v1", Vector3f(1, 0, 0));
        const auto v2 = parameters.get_vector3f("v2", Vector3f(0, 1, 0));

        const auto udelta = parameters.get_float("udelta", 0.0);
        const auto vdelta = parameters.get_float("vdelta", 0.0);

        *texture_mapping = PlanarMapping(render_from_texture.inverse(), v1, v2, udelta, vdelta);

        return texture_mapping;
    }

    if (type == "spherical") {
        *texture_mapping = SphericalMapping(render_from_texture.inverse());
        return texture_mapping;
    }

    if (type == "uv") {
        auto su = parameters.get_float("uscale", 1.);
        auto sv = parameters.get_float("vscale", 1.);
        auto du = parameters.get_float("udelta", 0.);
        auto dv = parameters.get_float("vdelta", 0.);

        *texture_mapping = UVMapping(su, sv, du, dv);

        return texture_mapping;
    }

    printf("ERROR: mapping `%s` not implemented\n", type.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}
