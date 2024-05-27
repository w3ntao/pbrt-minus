#include "pbrt/textures/spectrum_image_texture.h"
#include "pbrt/spectra/rgb_albedo_spectrum.h"

void SpectrumImageTexture::init(const ParameterDict &parameters, const RGBColorSpace *_color_space,
                                std::vector<void *> &gpu_dynamic_pointers) {
    auto image_path = parameters.root + "/" + parameters.get_string("filename", std::nullopt);

    spectrum_type = SpectrumType::Albedo;

    mapping = UVMapping(parameters);

    scale = parameters.get_float("scale", std::optional(1.0));
    invert = parameters.get_bool("invert", std::optional(false));

    auto max_anisotropy = parameters.get_float("maxanisotropy", std::optional(8.0));
    auto filter_string = parameters.get_string("filter", std::optional("bilinear"));

    auto mipmap_filter_options = MIPMapFilterOptions{
        .filter = parse_filter_function(filter_string),
        .max_anisotropy = max_anisotropy,
    };

    auto wrap_string = parameters.get_string("wrap", std::optional("repeat"));
    auto wrap_mode = parse_wrap_mode(wrap_string);

    MIPMap *_mipmap;
    CHECK_CUDA_ERROR(cudaMallocManaged(&_mipmap, sizeof(MIPMap)));
    _mipmap->init(mipmap_filter_options, wrap_mode, image_path, _color_space, gpu_dynamic_pointers);
    mipmap = _mipmap;

    gpu_dynamic_pointers.push_back(_mipmap);

    color_space = _color_space;
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
        rgb_albedo_spectrum.init(color_space, rgb.clamp(0.0, 1.0));
        return rgb_albedo_spectrum.sample(lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
