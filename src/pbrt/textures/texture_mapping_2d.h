#pragma once

#include <pbrt/base/spectrum_texture.h>
#include <pbrt/euclidean_space/point2.h>
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

struct TextureMapping2D {
    enum class Type {
        uv,
        planar,
    };

    static const TextureMapping2D *create(const Transform &renderFromTexture,
                                          const ParameterDictionary &parameters,
                                          GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    TexCoord2D map(const TextureEvalContext &ctx) const;

  private:
    Type type;
    const void *ptr;

    void init(const UVMapping *uv_mapping);
};
