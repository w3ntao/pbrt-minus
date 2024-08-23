#pragma once

#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/point2.h"
#include "pbrt/base/spectrum_texture.h"
#include "pbrt/scene/parameter_dictionary.h"

class UVMapping;

// TexCoord2D Definition
struct TexCoord2D {
    Point2f st;
    FloatType dsdx, dsdy, dtdx, dtdy;
};

struct TextureMapping2D {
    enum class Type {
        uv,
        planar,
    };

    Type type;
    const void *ptr;

    void init(const UVMapping *uv_mapping);

    PBRT_CPU_GPU
    TexCoord2D map(const TextureEvalContext &ctx) const;
};

// UVMapping Definition
class UVMapping {
  public:
    UVMapping(const ParameterDictionary &parameters) {
        this->init(parameters);
    }

    void init(const ParameterDictionary &parameters) {
        su = parameters.get_float("uscale", 1.0);
        sv = parameters.get_float("vscale", 1.0);
        du = parameters.get_float("udelta", 0.0);
        dv = parameters.get_float("vdelta", 0.0);
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
};
