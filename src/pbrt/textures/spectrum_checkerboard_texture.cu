#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/textures/spectrum_checkerboard_texture.h>
#include <pbrt/textures/texture_mapping_2d.h>
#include <pbrt/textures/texture_mapping_3d.h>

PBRT_CPU_GPU
static Real checkerboard(const TextureEvalContext &ctx, const TextureMapping2D *map2D,
                         const TextureMapping3D *map3D) {
    // Define 1D checkerboard filtered integral functions
    auto d = [](Real x) {
        Real y = x / 2 - std::floor(x / 2) - 0.5f;
        return x / 2 + y * (1 - 2 * std::abs(y));
    };

    auto bf = [&](Real x, Real r) -> Real {
        if (std::floor(x - r) == std::floor(x + r))
            return 1 - 2 * (int(std::floor(x)) & 1);
        return (d(x + r) - 2 * d(x) + d(x - r)) / sqr(r);
    };

    if (map2D) {
        TexCoord2D c = map2D->map(ctx);
        auto ds = std::max(std::abs(c.dsdx), std::abs(c.dsdy));
        auto dt = std::max(std::abs(c.dtdx), std::abs(c.dtdy));
        // Integrate product of 2D checkerboard function and triangle filter
        ds *= 1.5f;
        dt *= 1.5f;
        return 0.5f - bf(c.st[0], ds) * bf(c.st[1], dt) / 2;
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

SpectrumCheckerboardTexture::SpectrumCheckerboardTexture(const Transform &renderFromTexture,
                                                         SpectrumType spectrumType,
                                                         const ParameterDictionary &parameters,
                                                         GPUMemoryAllocator &allocator) {
    const auto dimension = parameters.get_integer("dimension", 2);
    if (dimension != 2 && dimension != 3) {
        REPORT_FATAL_ERROR();
    }

    auto tex1 = parameters.get_spectrum_texture("tex1", spectrumType, allocator);
    if (tex1 == nullptr) {
        const auto zero = Spectrum::create_constant_spectrum(0.0, allocator);
        tex1 = SpectrumTexture::create_constant_texture(zero, allocator);
    }

    auto tex2 = parameters.get_spectrum_texture("tex2", spectrumType, allocator);

    if (tex2 == nullptr) {
        const auto one = Spectrum::create_constant_spectrum(1.0, allocator);
        tex2 = SpectrumTexture::create_constant_texture(one, allocator);
    }

    if (dimension == 2) {
        const auto map2D = TextureMapping2D::create(renderFromTexture, parameters, allocator);
        init(map2D, nullptr, tex1, tex2);

        return;
    }

    std::cerr << "\nERROR: dimension 3 is not implemented yet\n";
    REPORT_FATAL_ERROR();
}

PBRT_CPU_GPU
SampledSpectrum SpectrumCheckerboardTexture::evaluate(const TextureEvalContext &ctx,
                                                      const SampledWavelengths &lambda) const {
    const auto w = checkerboard(ctx, map2D, map3D);

    SampledSpectrum t0, t1;
    if (w != 1) {
        t0 = tex0->evaluate(ctx, lambda);
    }

    if (w != 0) {
        t1 = tex1->evaluate(ctx, lambda);
    }

    return (1 - w) * t0 + w * t1;
}
