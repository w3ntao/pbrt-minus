#pragma once

#include <pbrt/base/bxdf_util.h>
#include <pbrt/spectrum_util/sampled_spectrum.h>
#include <pbrt/util/optional.h>

class DiffuseTransmissionBxDF {
  public:
    PBRT_CPU_GPU
    DiffuseTransmissionBxDF() : R(SampledSpectrum(NAN)), T(SampledSpectrum(NAN)) {}

    PBRT_CPU_GPU
    DiffuseTransmissionBxDF(const SampledSpectrum &R, const SampledSpectrum &T) : R(R), T(T) {}

    PBRT_CPU_GPU
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi, TransportMode mode) const {
        const auto InvPi = 1.0 / compute_pi();
        return wo.same_hemisphere(wi) ? (R * InvPi) : (T * InvPi);
    }

    PBRT_CPU_GPU
    pbrt::optional<BSDFSample>
    sample_f(const Vector3f &wo, const FloatType uc, const Point2f &u, const TransportMode mode,
             const BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        // Compute reflection and transmission probabilities for diffuse BSDF
        auto pr = R.max_component_value();
        auto pt = T.max_component_value();

        if (!(sampleFlags & BxDFReflTransFlags::Reflection)) {
            pr = 0;
        }
        if (!(sampleFlags & BxDFReflTransFlags::Transmission)) {
            pt = 0;
        }
        if (pr == 0 && pt == 0) {
            return {};
        }

        // Randomly sample diffuse BSDF reflection or transmission
        if (uc < pr / (pr + pt)) {
            // Sample diffuse BSDF reflection
            Vector3f wi = sample_cosine_hemisphere(u);
            if (wo.z < 0) {
                wi.z *= -1;
            }

            auto pdf = cosine_hemisphere_pdf(wi.abs_cos_theta()) * pr / (pr + pt);
            return BSDFSample(f(wo, wi, mode), wi, pdf, BxDFFlags::DiffuseReflection);
        }

        // Sample diffuse BSDF transmission
        Vector3f wi = sample_cosine_hemisphere(u);
        if (wo.z > 0) {
            wi.z *= -1;
        }

        auto pdf = cosine_hemisphere_pdf(wi.abs_cos_theta()) * pt / (pr + pt);
        return BSDFSample(f(wo, wi, mode), wi, pdf, BxDFFlags::DiffuseTransmission);
    }

    PBRT_CPU_GPU
    FloatType pdf(const Vector3f &wo, const Vector3f &wi, const TransportMode mode,
                  BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        // Compute reflection and transmission probabilities for diffuse BSDF
        auto pr = R.max_component_value();
        auto pt = T.max_component_value();
        if (!(sampleFlags & BxDFReflTransFlags::Reflection)) {
            pr = 0;
        }
        if (!(sampleFlags & BxDFReflTransFlags::Transmission)) {
            pt = 0;
        }
        if (pr == 0 && pt == 0) {
            return {};
        }

        return (wo.same_hemisphere(wi) ? pr : pt) / (pr + pt) *
               cosine_hemisphere_pdf(wi.abs_cos_theta());
    }

    PBRT_CPU_GPU
    void regularize() {}

    PBRT_CPU_GPU
    BxDFFlags flags() const {
        return (R.is_positive() ? BxDFFlags::DiffuseReflection : BxDFFlags::Unset) |
               (T.is_positive() ? BxDFFlags::DiffuseTransmission : BxDFFlags::Unset);
    }

  private:
    // DiffuseTransmissionBxDF Private Members
    SampledSpectrum R, T;
};
