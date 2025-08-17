#include <pbrt/textures/texture_eval_context.h>
#include <pbrt/textures/texture_mapping_2d.h>

PBRT_CPU_GPU
TexCoord2D CylindricalMapping::map(const TextureEvalContext &ctx) const {
    const Real Inv2Pi = 0.5 * pbrt::InvPI;

    Point3f pt = textureFromRender(ctx.p);
    // Compute texture coordinate differentials for cylinder $(u,v)$ mapping
    Real x2y2 = sqr(pt.x) + sqr(pt.y);
    Vector3f dsdp = Vector3f(-pt.y, pt.x, 0) / (2 * pbrt::PI * x2y2), dtdp = Vector3f(0, 0, 1);

    auto dpdx = textureFromRender(ctx.dpdx);
    auto dpdy = textureFromRender(ctx.dpdy);

    auto dsdx = dsdp.dot(dpdx);
    auto dsdy = dsdp.dot(dpdy);
    auto dtdx = dtdp.dot(dpdx);
    auto dtdy = dtdp.dot(dpdy);

    Point2f st((pbrt::PI + std::atan2(pt.y, pt.x)) * Inv2Pi, pt.z);
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

    const Real Inv2Pi = 0.5 * pbrt::InvPI;

    Real x2y2 = sqr(pt.x) + sqr(pt.y);
    Real sqrtx2y2 = std::sqrt(x2y2);
    Vector3f dsdp = Vector3f(-pt.y, pt.x, 0) / (2 * pbrt::PI * x2y2);
    Vector3f dtdp = 1 / (pbrt::PI * (x2y2 + sqr(pt.z))) *
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

    Point2f st(vec.spherical_theta() * pbrt::InvPI, vec.spherical_phi() * Inv2Pi);

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
    const auto type = parameters.get_one_string("mapping", "uv");
    if (type == "cylindrical") {
        const auto cylindrical_mapping =
            allocator.create<CylindricalMapping>(render_from_texture.inverse());

        return allocator.create<TextureMapping2D>(cylindrical_mapping);
    }

    if (type == "planar") {
        const auto v1 = parameters.get_vector3f("v1", Vector3f(1, 0, 0));
        const auto v2 = parameters.get_vector3f("v2", Vector3f(0, 1, 0));

        const auto udelta = parameters.get_float("udelta", 0.0);
        const auto vdelta = parameters.get_float("vdelta", 0.0);

        const auto planar_mapping =
            allocator.create<PlanarMapping>(render_from_texture.inverse(), v1, v2, udelta, vdelta);

        return allocator.create<TextureMapping2D>(planar_mapping);
    }

    if (type == "spherical") {
        const auto spherical_mapping =
            allocator.create<SphericalMapping>(render_from_texture.inverse());

        return allocator.create<TextureMapping2D>(spherical_mapping);
    }

    if (type == "uv") {
        auto su = parameters.get_float("uscale", 1.);
        auto sv = parameters.get_float("vscale", 1.);
        auto du = parameters.get_float("udelta", 0.);
        auto dv = parameters.get_float("vdelta", 0.);

        const auto uv_mapping = allocator.create<UVMapping>(su, sv, du, dv);

        return allocator.create<TextureMapping2D>(uv_mapping);
    }

    printf("ERROR: mapping `%s` not implemented\n", type.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

PBRT_CPU_GPU
TexCoord2D TextureMapping2D::map(const TextureEvalContext &ctx) const {
    switch (type) {
    case Type::cylindrical: {
        return static_cast<const CylindricalMapping *>(ptr)->map(ctx);
    }

    case Type::planar: {
        return static_cast<const PlanarMapping *>(ptr)->map(ctx);
    }

    case Type::spherical: {
        return static_cast<const SphericalMapping *>(ptr)->map(ctx);
    }

    case Type::uv: {
        return static_cast<const UVMapping *>(ptr)->map(ctx);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
