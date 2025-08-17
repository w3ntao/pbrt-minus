#include <pbrt/spectra/rgb_albedo_spectrum.h>
#include <pbrt/textures/mipmap.h>
#include <pbrt/textures/spectrum_image_texture.h>
#include <pbrt/textures/texture_mapping_2d.h>

SpectrumImageTexture::SpectrumImageTexture(SpectrumType _spectrum_type,
                                           const Transform &render_from_object,
                                           const RGBColorSpace *_color_space,
                                           const ParameterDictionary &parameters,
                                           GPUMemoryAllocator &allocator)
    : ImageTextureBase(render_from_object, parameters, allocator), spectrum_type(_spectrum_type),
      color_space(_color_space) {}

PBRT_CPU_GPU
SampledSpectrum SpectrumImageTexture::evaluate(const TextureEvalContext &ctx,
                                               const SampledWavelengths &lambda) const {
    auto c = texture_mapping->map(ctx);
    c.st[1] = 1.0 - c.st[1];

    auto _rgb = scale * mipmap->filter(c.st);
    auto rgb = (invert ? RGB(1.0, 1.0, 1.0) - _rgb : _rgb).clamp(0.0, Infinity);

    switch (spectrum_type) {
    case (SpectrumType::Albedo): {
        const auto rgb_albedo_spectrum = RGBAlbedoSpectrum(rgb.clamp(0.0, 1.0), color_space);
        return rgb_albedo_spectrum.sample(lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
