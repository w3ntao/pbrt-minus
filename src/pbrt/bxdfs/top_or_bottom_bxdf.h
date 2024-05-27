#include <cuda/std/optional>

#include "pbrt/util/macro.h"
#include "pbrt/base/bxdf.h"

template <typename TopBxDF, typename BottomBxDF>
class TopOrBottomBxDF {
  public:
    PBRT_CPU_GPU
    TopOrBottomBxDF() : top(nullptr), bottom(nullptr) {}

    PBRT_CPU_GPU
    TopOrBottomBxDF &operator=(const TopBxDF *t) {
        top = t;
        bottom = nullptr;
        return *this;
    }
    PBRT_CPU_GPU
    TopOrBottomBxDF &operator=(const BottomBxDF *b) {
        bottom = b;
        top = nullptr;
        return *this;
    }

    PBRT_CPU_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const {
        return top ? top->f(wo, wi, mode) : bottom->f(wo, wi, mode);
    }

    PBRT_CPU_GPU
    cuda::std::optional<BSDFSample>
    sample_f(Vector3f wo, FloatType uc, Point2f u, TransportMode mode,
             BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        return top ? top->sample_f(wo, uc, u, mode, sampleFlags)
                   : bottom->sample_f(wo, uc, u, mode, sampleFlags);
    }

    PBRT_CPU_GPU
    FloatType pdf(Vector3f wo, Vector3f wi, TransportMode mode,
                  BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        return top ? top->pdf(wo, wi, mode, sampleFlags) : bottom->pdf(wo, wi, mode, sampleFlags);
    }

    PBRT_CPU_GPU
    BxDFFlags flags() const {
        return top ? top->flags() : bottom->flags();
    }

  private:
    const TopBxDF *top = nullptr;
    const BottomBxDF *bottom = nullptr;
};
