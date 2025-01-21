#include <pbrt/textures/texture_mapping_2d.h>

const TextureMapping2D *TextureMapping2D::create(const Transform &render_from_texture,
                                                 const ParameterDictionary &parameters,
                                                 GPUMemoryAllocator &allocator) {
    auto texture_mapping = allocator.allocate<TextureMapping2D>();

    const auto type = parameters.get_one_string("mapping", "uv");
    if (type == "cylindrical") {
        auto spherical_mapping =
            CylindricalMapping::create(render_from_texture.inverse(), allocator);
        texture_mapping->init(spherical_mapping);

        return texture_mapping;
    }

    if (type == "planar") {
        const auto v1 = parameters.get_vector3f("v1", Vector3f(1, 0, 0));
        const auto v2 = parameters.get_vector3f("v2", Vector3f(0, 1, 0));

        const auto udelta = parameters.get_float("udelta", 0.0);
        const auto vdelta = parameters.get_float("vdelta", 0.0);

        auto planar_mapping =
            PlanarMapping::create(render_from_texture.inverse(), v1, v2, udelta, vdelta, allocator);

        texture_mapping->init(planar_mapping);

        return texture_mapping;
    }

    if (type == "spherical") {
        auto spherical_mapping = SphericalMapping::create(render_from_texture.inverse(), allocator);

        texture_mapping->init(spherical_mapping);

        return texture_mapping;
    }

    if (type == "uv") {
        auto su = parameters.get_float("uscale", 1.);
        auto sv = parameters.get_float("vscale", 1.);
        auto du = parameters.get_float("udelta", 0.);
        auto dv = parameters.get_float("vdelta", 0.);

        auto uv_mapping = UVMapping::create(su, sv, du, dv, allocator);

        texture_mapping->init(uv_mapping);

        return texture_mapping;
    }

    printf("ERROR: mapping `%s` not implemented\n", type.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

void TextureMapping2D::init(const CylindricalMapping *cylindrical_mapping) {
    type = Type::cylindrical;
    ptr = cylindrical_mapping;
}

void TextureMapping2D::init(const PlanarMapping *planar_mapping) {
    type = Type::planar;
    ptr = planar_mapping;
}

void TextureMapping2D::init(const SphericalMapping *spherical_mapping) {
    type = Type::spherical;
    ptr = spherical_mapping;
}

void TextureMapping2D::init(const UVMapping *uv_mapping) {
    type = Type::uv;
    ptr = uv_mapping;
}

PBRT_CPU_GPU
TexCoord2D TextureMapping2D::map(const TextureEvalContext &ctx) const {
    switch (type) {
    case Type::cylindrical: {
        return ((CylindricalMapping *)ptr)->map(ctx);
    }

    case Type::planar: {
        return ((PlanarMapping *)ptr)->map(ctx);
    }

    case Type::spherical: {
        return ((SphericalMapping *)ptr)->map(ctx);
    }

    case Type::uv: {
        return ((UVMapping *)ptr)->map(ctx);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
