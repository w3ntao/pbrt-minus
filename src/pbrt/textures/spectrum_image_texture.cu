#include "pbrt/textures/spectrum_image_texture.h"
#include "pbrt/spectra/rgb_albedo_spectrum.h"

const SpectrumImageTexture *
SpectrumImageTexture::create(const ParameterDictionary &parameters,
                             const RGBColorSpace *_color_space,
                             std::vector<void *> &gpu_dynamic_pointers) {
    SpectrumImageTexture *texture;
    CHECK_CUDA_ERROR(cudaMallocManaged(&texture, sizeof(SpectrumImageTexture)));
    texture->init(parameters, gpu_dynamic_pointers, _color_space);

    gpu_dynamic_pointers.push_back(texture);

    return texture;
}

void SpectrumImageTexture::init(const ParameterDictionary &parameters,
                                std::vector<void *> &gpu_dynamic_pointers,
                                const RGBColorSpace *_color_space) {
    color_space = _color_space;
    spectrum_type = SpectrumType::Albedo;

    init_image_texture_base(parameters, gpu_dynamic_pointers);
}

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
        rgb_albedo_spectrum.init(rgb.clamp(0.0, 1.0), color_space);
        return rgb_albedo_spectrum.sample(lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
