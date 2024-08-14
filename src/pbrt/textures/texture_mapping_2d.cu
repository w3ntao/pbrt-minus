#include "pbrt/textures/texture_mapping_2d.h"

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
