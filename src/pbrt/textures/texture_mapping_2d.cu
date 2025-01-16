#include <pbrt/textures/texture_mapping_2d.h>

const TextureMapping2D *TextureMapping2D::create(const Transform &renderFromTexture,
                                                 const ParameterDictionary &parameters,
                                                 GPUMemoryAllocator &allocator) {
    const auto type = parameters.get_one_string("mapping", "uv");
    if (type == "uv") {
        auto su = parameters.get_float("uscale", 1.);
        auto sv = parameters.get_float("vscale", 1.);
        auto du = parameters.get_float("udelta", 0.);
        auto dv = parameters.get_float("vdelta", 0.);

        auto uv_mapping = UVMapping::create(su, sv, du, dv, allocator);

        auto texture_mapping_2d = allocator.allocate<TextureMapping2D>();
        texture_mapping_2d->init(uv_mapping);

        return texture_mapping_2d;
    }

    printf("ERROR: mapping `%s` not implemented\n", type.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

void TextureMapping2D::init(const UVMapping *uv_mapping) {
    type = Type::uv;
    ptr = uv_mapping;
}

PBRT_CPU_GPU
TexCoord2D TextureMapping2D::map(const TextureEvalContext &ctx) const {
    switch (type) {
    case Type::uv: {
        return ((UVMapping *)ptr)->map(ctx);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
