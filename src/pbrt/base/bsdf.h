#pragma once

#include <cuda/std/optional>

#include "pbrt/base/bxdf.h"
#include "pbrt/euclidean_space/frame.h"
#include "pbrt/euclidean_space/normal3f.h"
#include "pbrt/euclidean_space/vector3.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"

class BSDF {
  public:
    PBRT_GPU void init_frame(const Normal3f &ns, const Vector3f &dpdus);

    PBRT_GPU void init_bxdf(CoatedConductorBxDF *coated_conductor_bxdf);

    PBRT_GPU void init_bxdf(CoatedDiffuseBxDF *coated_diffuse_bxdf);

    PBRT_GPU void init_bxdf(ConductorBxDF *conductor_bxdf);

    PBRT_GPU void init_bxdf(DielectricBxDF *dielectric_bxdf);

    PBRT_GPU void init_bxdf(DiffuseBxDF *diffuse_bxdf);

    PBRT_GPU
    void regularize() {
        bxdf.regularize();
    }

    PBRT_GPU
    SampledSpectrum f(const Vector3f &woRender, const Vector3f &wiRender,
                      TransportMode mode = TransportMode::Radiance) const;

    PBRT_GPU
    cuda::std::optional<BSDFSample>
    sample_f(const Vector3f &wo_render, FloatType u, const Point2f &u2,
             TransportMode mode = TransportMode::Radiance,
             BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const;

    PBRT_GPU
    FloatType pdf(const Vector3f &woRender, const Vector3f &wiRender,
                  TransportMode mode = TransportMode::Radiance,
                  BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    PBRT_CPU_GPU BxDFFlags flags() const {
        return bxdf.flags();
    }

    PBRT_CPU_GPU
    bool has_null_bxdf() const {
        return bxdf.has_type_null();
    }

  private:
    PBRT_CPU_GPU
    Vector3f render_to_local(const Vector3f &v) const {
        return shading_frame.to_local(v);
    }

    PBRT_CPU_GPU
    Vector3f local_to_render(const Vector3f &v) const {
        return shading_frame.from_local(v);
    }

    Frame shading_frame;
    BxDF bxdf;
};
