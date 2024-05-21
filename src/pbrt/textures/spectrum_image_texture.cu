#include "pbrt/textures/spectrum_image_texture.h"
#include "pbrt/spectra/rgb_albedo_spectrum.h"

PBRT_CPU_GPU
SampledSpectrum SpectrumImageTexture::evaluate(const TextureEvalContext &ctx,
                                               const SampledWavelengths &lambda) const {
    auto c = mapping.map(ctx);
    c.st[1] = 1.0 - c.st[1];

    auto _rgb = scale * mipmap->filter(c.st);
    auto rgb = (invert ? RGB(1.0, 1.0, 1.0) - _rgb : _rgb).clamp(0.0, Infinity);

    switch (spectrum_type) {
    case (SpectrumType::Albedo): {
        RGBAlbedoSpectrum rgb_albedo_spectrum;
        rgb_albedo_spectrum.init(color_space, rgb.clamp(0.0, 1.0));
        return rgb_albedo_spectrum.sample(lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
