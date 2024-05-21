#pragma once

#include "pbrt/base/spectrum.h"
#include "pbrt/euclidean_space/transform.h"

#include "pbrt/textures/image_texture_base.h"
#include "pbrt/textures/mipmap.h"

class SpectrumImageTexture : ImageTextureBase {
  public:
    void new_init(const ParameterDict &parameters, const RGBColorSpace *_color_space,
                  std::vector<void *> &gpu_dynamic_pointers) {
        auto image_path =
            parameters.file_root + "/" + parameters.get_string("filename", std::nullopt);

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
        _mipmap->init(mipmap_filter_options, wrap_mode, image_path, _color_space,
                      gpu_dynamic_pointers);
        mipmap = _mipmap;

        gpu_dynamic_pointers.push_back(_mipmap);

        color_space = _color_space;
    }

    void init(const ParameterDict &parameters, const std::string &image_path,
              const RGBColorSpace *_color_space, std::vector<void *> &gpu_dynamic_pointers) {
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
        _mipmap->init(mipmap_filter_options, wrap_mode, image_path, _color_space,
                      gpu_dynamic_pointers);
        mipmap = _mipmap;

        gpu_dynamic_pointers.push_back(_mipmap);

        color_space = _color_space;
    }

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    SpectrumType spectrum_type;
    const RGBColorSpace *color_space;
};
