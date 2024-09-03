#pragma once

#include "pbrt/base/media.h"
#include "pbrt/bxdfs/top_or_bottom_bxdf.h"
#include "pbrt/util/hash.h"
#include "pbrt/util/macro.h"
#include <curand_kernel.h>

// LayeredBxDF Definition
template <typename TopBxDF, typename BottomBxDF, bool twoSided>
class LayeredBxDF {
  public:
    PBRT_GPU
    LayeredBxDF() {};

    PBRT_GPU
    LayeredBxDF(TopBxDF top, BottomBxDF bottom, FloatType thickness, const SampledSpectrum &albedo,
                FloatType g, int maxDepth, int nSamples)
        : top(top), bottom(bottom),
          thickness(std::max(thickness, std::numeric_limits<FloatType>::min())), g(g),
          albedo(albedo), maxDepth(maxDepth), nSamples(nSamples) {}

    PBRT_GPU
    void regularize() {
        top.regularize();
        bottom.regularize();
    }

    PBRT_CPU_GPU
    BxDFFlags flags() const {
        BxDFFlags topFlags = top.flags();
        BxDFFlags bottomFlags = bottom.flags();

        BxDFFlags flags = BxDFFlags::Reflection;
        if (_is_specular(topFlags)) {
            flags = flags | BxDFFlags::Specular;
        }

        if (_is_diffuse(topFlags) || _is_diffuse(bottomFlags) || albedo.is_positive()) {
            flags = flags | BxDFFlags::Diffuse;
        } else if (_is_glossy(topFlags) || _is_glossy(bottomFlags)) {
            flags = flags | BxDFFlags::Glossy;
        }

        if (_is_transmissive(topFlags) && _is_transmissive(bottomFlags)) {
            flags = flags | BxDFFlags::Transmission;
        }

        return flags;
    }

    PBRT_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const {
        SampledSpectrum f(0.);
        // Estimate _LayeredBxDF_ value _f_ using random sampling
        // Set _wo_ and _wi_ for layered BSDF evaluation
        if (twoSided && wo.z < 0) {
            wo = -wo;
            wi = -wi;
        }

        // Determine entrance interface for layered BSDF
        TopOrBottomBxDF<TopBxDF, BottomBxDF> enterInterface;
        bool enteredTop = twoSided || wo.z > 0;
        if (enteredTop) {
            enterInterface = &top;
        } else {
            enterInterface = &bottom;
        }

        // Determine exit interface and exit $z$ for layered BSDF
        TopOrBottomBxDF<TopBxDF, BottomBxDF> exitInterface, nonExitInterface;
        if (wo.same_hemisphere(wi) ^ enteredTop) {
            exitInterface = &bottom;
            nonExitInterface = &top;
        } else {
            exitInterface = &top;
            nonExitInterface = &bottom;
        }

        FloatType exitZ = (wo.same_hemisphere(wi) ^ enteredTop) ? 0 : thickness;

        // Account for reflection at the entrance interface
        if (wo.same_hemisphere(wi)) {
            f = nSamples * enterInterface.f(wo, wi, mode);
        }

        curandState rand_state;
        curand_init(pstd::hash(wi), pstd::hash(wo), 0, &rand_state);
        auto r = [&rand_state]() {
            return std::min<FloatType>(curand_uniform(&rand_state), OneMinusEpsilon);
        };

        for (int s = 0; s < nSamples; ++s) {
            // Sample random walk through layers to estimate BSDF value
            // Sample transmission direction through entrance interface
            FloatType uc = r();
            auto wos = enterInterface.sample_f(wo, uc, Point2f(r(), r()), mode,
                                               BxDFReflTransFlags::Transmission);
            if (!wos || !wos->f.is_positive() || wos->pdf == 0 || wos->wi.z == 0) {
                continue;
            }

            // Sample BSDF for virtual light from _wi_
            uc = r();
            auto wis = exitInterface.sample_f(wi, uc, Point2f(r(), r()), !mode,
                                              BxDFReflTransFlags::Transmission);

            if (!wis || !wis->f.is_positive() || wis->pdf == 0 || wis->wi.z == 0) {
                continue;
            }

            // Declare state for random walk through BSDF layers
            // SampledSpectrum beta = wos->f * AbsCosTheta(wos->wi) / wos->pdf;

            SampledSpectrum beta = wos->f * wos->wi.abs_cos_theta() / wos->pdf;

            FloatType z = enteredTop ? thickness : 0;
            Vector3f w = wos->wi;

            HGPhaseFunction phase(g);

            for (int depth = 0; depth < maxDepth; ++depth) {
                // Sample next event for layered BSDF evaluation random walk

                // Possibly terminate layered BSDF random walk with Russian roulette
                if (depth > 3 && beta.max_component_value() < 0.25f) {
                    FloatType q = std::max<FloatType>(0, 1 - beta.max_component_value());
                    if (r() < q)
                        break;
                    beta /= 1 - q;
                }

                // Account for media between layers and possibly scatter
                if (!albedo.is_positive()) {
                    // Advance to next layer boundary and update _beta_ for transmittance
                    z = (z == thickness) ? 0 : thickness;
                    beta *= Tr(thickness, w);
                } else {
                    // Sample medium scattering for layered BSDF evaluation
                    FloatType sigma_t = 1;
                    FloatType dz = SampleExponential(r(), sigma_t / std::abs(w.z));
                    FloatType zp = w.z > 0 ? (z + dz) : (z - dz);

                    if (z == zp) {
                        continue;
                    }

                    if (0 < zp && zp < thickness) {
                        // Handle scattering event in layered BSDF medium
                        // Account for scattering through _exitInterface_ using _wis_
                        FloatType wt = 1;
                        if (!_is_specular(exitInterface.flags())) {
                            wt = power_heuristic(1, wis->pdf, 1, phase.pdf(-w, -wis->wi));
                        }

                        f += beta * albedo * phase.p(-w, -wis->wi) * wt * Tr(zp - exitZ, wis->wi) *
                             wis->f / wis->pdf;

                        // Sample phase function and update layered path state
                        Point2f u{r(), r()};
                        cuda::std::optional<PhaseFunctionSample> ps = phase.sample_p(-w, u);
                        if (!ps || ps->pdf == 0 || ps->wi.z == 0) {
                            continue;
                        }

                        beta *= albedo * ps->p / ps->pdf;
                        w = ps->wi;
                        z = zp;

                        // Possibly account for scattering through _exitInterface_
                        if (((z < exitZ && w.z > 0) || (z > exitZ && w.z < 0)) &&
                            !_is_specular(exitInterface.flags())) {
                            // Account for scattering through _exitInterface_
                            SampledSpectrum fExit = exitInterface.f(-w, wi, mode);
                            if (fExit.is_positive()) {
                                FloatType exitPDF = exitInterface.pdf(
                                    -w, wi, mode, BxDFReflTransFlags::Transmission);
                                FloatType wt = power_heuristic(1, ps->pdf, 1, exitPDF);
                                f += beta * Tr(zp - exitZ, ps->wi) * fExit * wt;
                            }
                        }

                        continue;
                    }
                    z = clamp<FloatType>(zp, 0, thickness);
                }

                // Account for scattering at appropriate interface
                if (z == exitZ) {
                    // Account for reflection at _exitInterface_
                    FloatType uc = r();
                    cuda::std::optional<BSDFSample> bs = exitInterface.sample_f(
                        -w, uc, Point2f(r(), r()), mode, BxDFReflTransFlags::Reflection);
                    if (!bs || !bs->f.is_positive() || bs->pdf == 0 || bs->wi.z == 0) {
                        break;
                    }

                    beta *= bs->f * bs->wi.abs_cos_theta() / bs->pdf;
                    w = bs->wi;

                } else {
                    // Account for scattering at _nonExitInterface_
                    if (!_is_specular(nonExitInterface.flags())) {
                        // Add NEE contribution along presampled _wis_ direction
                        FloatType wt = 1;
                        if (!_is_specular(exitInterface.flags())) {
                            wt = power_heuristic(1, wis->pdf, 1,
                                                 nonExitInterface.pdf(-w, -wis->wi, mode));
                        }

                        f += beta * nonExitInterface.f(-w, -wis->wi, mode) *
                             wis->wi.abs_cos_theta() * wt * Tr(thickness, wis->wi) * wis->f /
                             wis->pdf;
                    }
                    // Sample new direction using BSDF at _nonExitInterface_
                    FloatType uc = r();
                    Point2f u(r(), r());
                    cuda::std::optional<BSDFSample> bs =
                        nonExitInterface.sample_f(-w, uc, u, mode, BxDFReflTransFlags::Reflection);
                    if (!bs || !bs->f.is_positive() || bs->pdf == 0 || bs->wi.z == 0) {
                        break;
                    }

                    beta *= bs->f * bs->wi.abs_cos_theta() / bs->pdf;
                    w = bs->wi;

                    if (!_is_specular(exitInterface.flags())) {
                        // Add NEE contribution along direction from BSDF sample
                        SampledSpectrum fExit = exitInterface.f(-w, wi, mode);
                        if (fExit.is_positive()) {
                            FloatType wt = 1;
                            if (!_is_specular(nonExitInterface.flags())) {
                                FloatType exitPDF = exitInterface.pdf(
                                    -w, wi, mode, BxDFReflTransFlags::Transmission);
                                wt = power_heuristic(1, bs->pdf, 1, exitPDF);
                            }
                            f += beta * Tr(thickness, bs->wi) * fExit * wt;
                        }
                    }
                }
            }
        }

        return f / nSamples;
    }

    PBRT_GPU
    cuda::std::optional<BSDFSample>
    sample_f(Vector3f wo, FloatType uc, Point2f u, TransportMode mode,
             BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        // Set _wo_ for layered BSDF sampling
        bool flipWi = false;
        if (twoSided && wo.z < 0) {
            wo = -wo;
            flipWi = true;
        }

        // Sample BSDF at entrance interface to get initial direction _w_
        bool enteredTop = twoSided || wo.z > 0;
        cuda::std::optional<BSDFSample> bs =
            enteredTop ? top.sample_f(wo, uc, u, mode) : bottom.sample_f(wo, uc, u, mode);
        if (!bs || !bs->f.is_positive() || bs->pdf == 0 || bs->wi.z == 0) {
            return {};
        }

        if (bs->is_reflection()) {
            if (flipWi) {
                bs->wi = -bs->wi;
            }

            bs->pdf_is_proportional = true;
            return bs;
        }
        Vector3f w = bs->wi;
        bool specularPath = bs->is_specular();

        curandState rand_state;
        curand_init(pstd::hash(wo), pstd::hash(uc, u), 0, &rand_state);

        auto r = [&rand_state]() {
            return std::min<FloatType>(curand_uniform(&rand_state), OneMinusEpsilon);
        };

        // Declare common variables for layered BSDF sampling
        SampledSpectrum f = bs->f * bs->wi.abs_cos_theta();
        FloatType pdf = bs->pdf;
        FloatType z = enteredTop ? thickness : 0;
        HGPhaseFunction phase(g);

        for (int depth = 0; depth < maxDepth; ++depth) {
            // Follow random walk through layers to sample layered BSDF
            // Possibly terminate layered BSDF sampling with Russian Roulette
            FloatType rrBeta = f.max_component_value() / pdf;
            if (depth > 3 && rrBeta < 0.25f) {
                FloatType q = std::max<FloatType>(0, 1 - rrBeta);
                if (r() < q) {
                    return {};
                }

                pdf *= 1 - q;
            }
            if (w.z == 0) {
                return {};
            }

            if (albedo.is_positive()) {
                // Sample potential scattering event in layered medium
                FloatType sigma_t = 1;
                FloatType dz = SampleExponential(r(), sigma_t / w.abs_cos_theta());
                FloatType zp = w.z > 0 ? (z + dz) : (z - dz);
                if (zp == z) {
                    return {};
                }

                if (0 < zp && zp < thickness) {
                    // Update path state for valid scattering event between interfaces
                    cuda::std::optional<PhaseFunctionSample> ps =
                        phase.sample_p(-w, Point2f(r(), r()));
                    if (!ps || ps->pdf == 0 || ps->wi.z == 0) {
                        return {};
                    }

                    f *= albedo * ps->p;
                    pdf *= ps->pdf;
                    specularPath = false;
                    w = ps->wi;
                    z = zp;

                    continue;
                }

                z = clamp<FloatType>(zp, 0, thickness);

            } else {
                // Advance to the other layer interface
                z = (z == thickness) ? 0 : thickness;
                f *= Tr(thickness, w);
            }
            // Initialize _interface_ for current interface surface
#ifdef interface // That's enough out of you, Windows.
#undef interface
#endif
            TopOrBottomBxDF<TopBxDF, BottomBxDF> interface;
            if (z == 0) {
                interface = &bottom;
            } else {
                interface = &top;
            }

            // Sample interface BSDF to determine new path direction
            FloatType uc = r();
            Point2f u(r(), r());
            cuda::std::optional<BSDFSample> bs = interface.sample_f(-w, uc, u, mode);
            if (!bs || !bs->f.is_positive() || bs->pdf == 0 || bs->wi.z == 0) {
                return {};
            }

            f *= bs->f;
            pdf *= bs->pdf;
            specularPath &= bs->is_specular();
            w = bs->wi;

            // Return _BSDFSample_ if path has left the layers
            if (bs->is_transmission()) {
                BxDFFlags flags =
                    wo.same_hemisphere(w) ? BxDFFlags::Reflection : BxDFFlags::Transmission;

                flags |= specularPath ? BxDFFlags::Specular : BxDFFlags::Glossy;
                if (flipWi) {
                    w = -w;
                }

                return BSDFSample(f, w, pdf, flags, 1.f, true);
            }

            // Scale _f_ by cosine term after scattering at the interface
            f *= bs->wi.abs_cos_theta();
        }
        return {};
    }

    PBRT_GPU
    FloatType pdf(Vector3f wo, Vector3f wi, TransportMode mode,
                  BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        // Set _wo_ and _wi_ for layered BSDF evaluation
        if (twoSided && wo.z < 0) {
            wo = -wo;
            wi = -wi;
        }

        curandState rand_state;
        curand_init(pstd::hash(wi), pstd::hash(wo), 0, &rand_state);
        auto r = [&rand_state]() {
            return std::min<FloatType>(curand_uniform(&rand_state), OneMinusEpsilon);
        };

        // Update _pdfSum_ for reflection at the entrance layer
        bool enteredTop = twoSided || wo.z > 0;
        FloatType pdfSum = 0;
        if (wo.same_hemisphere(wi)) {
            auto reflFlag = BxDFReflTransFlags::Reflection;
            pdfSum += enteredTop ? nSamples * top.pdf(wo, wi, mode, reflFlag)
                                 : nSamples * bottom.pdf(wo, wi, mode, reflFlag);
        }

        for (int s = 0; s < nSamples; ++s) {
            // Evaluate layered BSDF pdf sample
            if (wo.same_hemisphere(wi)) {
                // Evaluate TRT term for pdf estimate
                TopOrBottomBxDF<TopBxDF, BottomBxDF> rInterface, tInterface;
                if (enteredTop) {
                    rInterface = &bottom;
                    tInterface = &top;
                } else {
                    rInterface = &top;
                    tInterface = &bottom;
                }
                // Sample _tInterface_ to get direction into the layers
                auto trans = BxDFReflTransFlags::Transmission;
                cuda::std::optional<BSDFSample> wos, wis;
                wos = tInterface.sample_f(wo, r(), {r(), r()}, mode, trans);
                wis = tInterface.sample_f(wi, r(), {r(), r()}, !mode, trans);

                // Update _pdfSum_ accounting for TRT scattering events
                if (wos && wos->f.is_positive() && wos->pdf > 0 && wis && wis->f.is_positive() &&
                    wis->pdf > 0) {
                    if (!_is_non_specular(tInterface.flags())) {
                        pdfSum += rInterface.pdf(-wos->wi, -wis->wi, mode);
                    } else {
                        // Use multiple importance sampling to estimate pdf product
                        cuda::std::optional<BSDFSample> rs =
                            rInterface.sample_f(-wos->wi, r(), {r(), r()}, mode);
                        if (rs && rs->f.is_positive() && rs->pdf > 0) {
                            if (!_is_non_specular(rInterface.flags()))
                                pdfSum += tInterface.pdf(-rs->wi, wi, mode);
                            else {
                                // Compute MIS-weighted estimate of Equation
                                // (\ref{eq:pdf-triple-canceled-one})
                                FloatType rPDF = rInterface.pdf(-wos->wi, -wis->wi, mode);
                                FloatType wt = power_heuristic(1, wis->pdf, 1, rPDF);
                                pdfSum += wt * rPDF;

                                FloatType tPDF = tInterface.pdf(-rs->wi, wi, mode);
                                wt = power_heuristic(1, rs->pdf, 1, tPDF);
                                pdfSum += wt * tPDF;
                            }
                        }
                    }
                }

            } else {
                // Evaluate TT term for pdf estimate
                TopOrBottomBxDF<TopBxDF, BottomBxDF> toInterface, tiInterface;
                if (enteredTop) {
                    toInterface = &top;
                    tiInterface = &bottom;
                } else {
                    toInterface = &bottom;
                    tiInterface = &top;
                }

                FloatType uc = r();
                Point2f u(r(), r());
                cuda::std::optional<BSDFSample> wos = toInterface.sample_f(wo, uc, u, mode);
                if (!wos || !wos->f.is_positive() || wos->pdf == 0 || wos->wi.z == 0 ||
                    wos->is_reflection()) {
                    continue;
                }

                uc = r();
                u = Point2f(r(), r());
                cuda::std::optional<BSDFSample> wis = tiInterface.sample_f(wi, uc, u, !mode);
                if (!wis || !wis->f.is_positive() || wis->pdf == 0 || wis->wi.z == 0 ||
                    wis->is_reflection()) {
                    continue;
                }

                if (_is_specular(toInterface.flags())) {
                    pdfSum += tiInterface.pdf(-wos->wi, wi, mode);
                } else if (_is_specular(tiInterface.flags())) {
                    pdfSum += toInterface.pdf(wo, -wis->wi, mode);
                } else {
                    pdfSum += (toInterface.pdf(wo, -wis->wi, mode) +
                               tiInterface.pdf(-wos->wi, wi, mode)) /
                              2;
                }
            }
        }
        // Return mixture of pdf estimate and constant pdf
        return lerp(0.9f, 1 / (4 * compute_pi()), pdfSum / nSamples);
    }

  private:
    // LayeredBxDF Private Methods
    PBRT_GPU
    static FloatType Tr(FloatType dz, Vector3f w) {
        if (std::abs(dz) <= std::numeric_limits<FloatType>::min()) {
            return 1;
        }

        return exp(-std::abs(dz / w.z));
    }

    // LayeredBxDF Private Members
    TopBxDF top;
    BottomBxDF bottom;
    FloatType thickness, g;
    SampledSpectrum albedo;
    int maxDepth, nSamples;
};
