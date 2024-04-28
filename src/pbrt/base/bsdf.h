#pragma once

#include "pbrt/base/bxdf.h"
#include "pbrt/euclidean_space/normal3f.h"
#include "pbrt/euclidean_space/frame.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"

class DiffuseBxDF;

class BSDF {
  public:
    PBRT_GPU void init_frame(const Normal3f &ns, const Vector3f &dpdus);

    PBRT_GPU void init_bxdf(const DiffuseBxDF *diffuse_bxdf);

    PBRT_GPU
    SampledSpectrum f(const Vector3f &woRender, const Vector3f &wiRender,
                      TransportMode mode = TransportMode::Radiance) const;

    PBRT_GPU
    bool has_null_bxdf() const {
        return bxdf.has_type_null();
    }

  private:
    PBRT_CPU_GPU
    Vector3f RenderToLocal(const Vector3f &v) const {
        return shading_frame.to_local(v);
    }

    PBRT_CPU_GPU
    Vector3f LocalToRender(const Vector3f &v) const {
        return shading_frame.from_local(v);
    }

    PBRT_CPU_GPU
    void report_error() const {
        const char *error_msg = "\nBSDF: this type is not implemented\n\n";

        printf(error_msg);
#if defined(__CUDA_ARCH__)
        asm("trap;");
#else
        throw std::runtime_error(error_msg);
#endif
    }

    Frame shading_frame;
    BxDF bxdf;
};
