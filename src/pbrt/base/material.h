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
    void init(const DiffuseMaterial *diffuse_material);

    PBRT_GPU
    DiffuseBxDF get_diffuse_bsdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

  private:
    enum class MaterialType {
        diffuse_material,
    };

    void *material_ptr;
    MaterialType material_type;

    PBRT_CPU_GPU
    void report_error() const {
        const char *error_msg = "\nMaterial: this type is not implemented\n\n";

        printf("%s", error_msg);
#if defined(__CUDA_ARCH__)
        asm("trap;");
#else
        throw std::runtime_error(error_msg);
#endif
    }
};
