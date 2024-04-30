#pragma once

#include <optional>

#include "pbrt/util/macro.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"

#include "pbrt/base/bsdf.h"
#include "pbrt/base/texture.h"
#include "pbrt/euclidean_space/vector3.h"
#include "pbrt/euclidean_space/normal3f.h"

struct MaterialEvalContext : public TextureEvalContext {
    // MaterialEvalContext Public Methods
    MaterialEvalContext() = default;

    PBRT_CPU_GPU
    MaterialEvalContext(const SurfaceInteraction &si)
        : TextureEvalContext(si), wo(si.wo), ns(si.shading.n), dpdus(si.shading.dpdu) {}

    Vector3f wo;
    Normal3f ns;
    Vector3f dpdus;
};

class DiffuseMaterial;

class Material {
  public:
    enum class MaterialType {
        diffuse_material,
    };

    void init(const DiffuseMaterial *diffuse_material);

    PBRT_GPU
    DiffuseBxDF get_diffuse_bsdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    MaterialType get_material_type() const {
        return material_type;
    }

  private:
    void *material_ptr;
    MaterialType material_type;
};
