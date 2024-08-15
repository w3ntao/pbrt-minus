#include "pbrt/textures/spectrum_image_texture.h"
#include "pbrt/spectra/rgb_albedo_spectrum.h"

const SpectrumImageTexture *
SpectrumImageTexture::create(SpectrumType spectrum_type, const Transform &render_from_object,
                             const RGBColorSpace *_color_space,
                             const ParameterDictionary &parameters,
                             std::vector<void *> &gpu_dynamic_pointers) {
    SpectrumImageTexture *texture;
    CHECK_CUDA_ERROR(cudaMallocManaged(&texture, sizeof(SpectrumImageTexture)));
    texture->init(spectrum_type, render_from_object, parameters, gpu_dynamic_pointers,
                  _color_space);

    gpu_dynamic_pointers.push_back(texture);

    return texture;
}

void SpectrumImageTexture::init(SpectrumType _spectrum_type, const Transform &render_from_object,
                                const ParameterDictionary &parameters,
                                std::vector<void *> &gpu_dynamic_pointers,
                                const RGBColorSpace *_color_space) {
    spectrum_type = _spectrum_type;
    color_space = _color_space;

    init_image_texture_base(render_from_object, parameters, gpu_dynamic_pointers);
}

PBRT_CPU_GPU
SampledSpectrum SpectrumImageTexture::evaluate(const TextureEvalContext &ctx,
                                               const SampledWavelengths &lambda) const {
    auto c = texture_mapping->map(ctx);
    c.st[1] = 1.0 - c.st[1];

    auto _rgb = scale * mipmap->filter(c.st);
    auto rgb = (invert ? RGB(1.0, 1.0, 1.0) - _rgb : _rgb).clamp(0.0, Infinity);

    switch (spectrum_type) {
    case (SpectrumType::Albedo): {
        RGBAlbedoSpectrum rgb_albedo_spectrum;
        rgb_albedo_spectrum.init(rgb.clamp(0.0, 1.0), color_space);
        return rgb_albedo_spectrum.sample(lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
