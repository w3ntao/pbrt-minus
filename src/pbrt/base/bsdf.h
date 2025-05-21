#pragma once

#include <pbrt/base/bxdf.h>
#include <pbrt/euclidean_space/frame.h>
#include <pbrt/euclidean_space/normal3f.h>

class Material;
class MaterialEvalContext;

class BSDF {
  public:
    PBRT_CPU_GPU
    BSDF() {}

    PBRT_CPU_GPU
    BSDF(const Normal3f &ns, const Vector3f &dpdus)
        : shading_frame(Frame::from_xz(dpdus.normalize(), ns.to_vector3())) {}

    PBRT_CPU_GPU
    void init_bxdf(const Material *material, SampledWavelengths &lambda,
                   const MaterialEvalContext &material_eval_context);

    PBRT_CPU_GPU
    void regularize() {
        bxdf.regularize();
    }

    PBRT_CPU_GPU
    SampledSpectrum f(const Vector3f &woRender, const Vector3f &wiRender,
                      TransportMode mode = TransportMode::Radiance) const;

    PBRT_CPU_GPU
    pbrt::optional<BSDFSample>
    sample_f(const Vector3f &wo_render, Real u, const Point2f &u2,
             TransportMode mode = TransportMode::Radiance,
             BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const;

    PBRT_CPU_GPU
    Real pdf(const Vector3f &woRender, const Vector3f &wiRender,
             TransportMode mode = TransportMode::Radiance,
             BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    PBRT_CPU_GPU BxDFFlags flags() const {
        return bxdf.flags();
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
