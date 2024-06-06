#pragma once

#include <vector>

#include "pbrt/util/macro.h"

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/point3.h"
#include "pbrt/euclidean_space/vector3.h"
#include "pbrt/euclidean_space/normal3f.h"

#include "pbrt/base/interaction.h"

class FloatConstantTexture;

class Spectrum;
class SpectrumConstantTexture;
class SpectrumImageTexture;
class SpectrumScaleTexture;

struct TextureEvalContext {
    Point3f p;
    Vector3f dpdx;
    Vector3f dpdy;

    Normal3f n;
    Point2f uv;

    FloatType dudx = 0;
    FloatType dudy = 0;
    FloatType dvdx = 0;
    FloatType dvdy = 0;

    int faceIndex = 0;

    TextureEvalContext() = default;

    PBRT_CPU_GPU
    explicit TextureEvalContext(const Interaction &intr) : p(intr.p()), uv(intr.uv) {}

    PBRT_CPU_GPU
    explicit TextureEvalContext(const SurfaceInteraction &si)
        : p(si.p()), dpdx(si.dpdx), dpdy(si.dpdy), n(si.n), uv(si.uv), dudx(si.dudx), dudy(si.dudy),
          dvdx(si.dvdx), dvdy(si.dvdy), faceIndex(si.faceIndex) {}
};

class FloatTexture {
  public:
    enum class Type {
        constant,
    };

    static const FloatTexture *create(FloatType val, std::vector<void *> &gpu_dynamic_pointers);

    void init(const FloatConstantTexture *constant_texture);

    PBRT_CPU_GPU
    FloatType evaluate(const TextureEvalContext &ctx) const;

  private:
    Type type;
    const void *ptr;
};

class SpectrumTexture {
  public:
    enum class Type {
        constant,
        image,
        scale,
    };

    static const SpectrumTexture *
    create_constant_float_val_texture(FloatType val, std::vector<void *> &gpu_dynamic_pointers);

    static const SpectrumTexture *
    create_constant_texture(const Spectrum *spectrum, std::vector<void *> &gpu_dynamic_pointers);

    void init(const SpectrumConstantTexture *constant_texture);

    void init(const SpectrumImageTexture *image_texture);

    void init(const SpectrumScaleTexture *scale_texture);

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    Type type;
    const void *ptr;
};
