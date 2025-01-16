#include <pbrt/base/spectrum_texture.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/textures/spectrum_checkerboard_texture.h>

PBRT_CPU_GPU
static FloatType checkerboard(const TextureEvalContext &ctx, const TextureMapping2D *map2D,
                              const TextureMapping3D *map3D) {
    // Define 1D checkerboard filtered integral functions
    auto d = [](FloatType x) {
        FloatType y = x / 2 - std::floor(x / 2) - 0.5f;
        return x / 2 + y * (1 - 2 * std::abs(y));
    };

    auto bf = [&](FloatType x, FloatType r) -> FloatType {
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

const SpectrumCheckerboardTexture *
SpectrumCheckerboardTexture::create(const Transform &renderFromTexture, SpectrumType spectrumType,
                                    const ParameterDictionary &parameters,
                                    GPUMemoryAllocator &allocator) {
    const auto dimension = parameters.get_integer("dimension", 2);
    if (dimension != 2 && dimension != 3) {
        REPORT_FATAL_ERROR();
    }

    const auto zero = Spectrum::create_constant_spectrum(0.0, allocator);
    const auto one = Spectrum::create_constant_spectrum(1.0, allocator);

    auto tex1 = parameters.get_spectrum_texture("tex1", spectrumType, allocator);
    if (tex1 == nullptr) {
        tex1 = SpectrumTexture::create_constant_texture(zero, allocator);
    }

    auto tex2 = parameters.get_spectrum_texture("tex2", spectrumType, allocator);

    if (tex2 == nullptr) {
        tex2 = SpectrumTexture::create_constant_texture(one, allocator);
    }

    auto checkerboard_texture = allocator.allocate<SpectrumCheckerboardTexture>();
    if (dimension == 2) {
        auto map2D = TextureMapping2D::create(renderFromTexture, parameters, allocator);
        checkerboard_texture->init(map2D, nullptr, tex1, tex2);

        return checkerboard_texture;
    }

    std::cerr << "\nERROR: dimension 3 is not implemented yet\n";
    REPORT_FATAL_ERROR();

    return nullptr;
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
