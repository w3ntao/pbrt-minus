#include "pbrt/base/float_texture.h"
#include "pbrt/base/spectrum.h"
#include "pbrt/base/spectrum_texture.h"
#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/scene/tokenizer.h"
#include "pbrt/spectra/black_body_spectrum.h"
#include "pbrt/spectra/rgb_illuminant_spectrum.h"
#include "pbrt/spectrum_util/global_spectra.h"

std::vector<FloatType> read_spectrum_file(const std::string &filename) {
    std::string token;
    std::ifstream file(filename);

    std::vector<FloatType> values;

    while (std::getline(file, token)) {
        std::istringstream line(token);
        while (line >> token) {
            values.push_back(std::stod(token));
        }
        if (file.unget().get() == '\n') {
            // if it's '\n', do nothing
        }
    }

    return values;
}

ParameterDictionary::ParameterDictionary(
    const std::vector<Token> &tokens, const std::string &_root,
    const GlobalSpectra *_global_spectra, const std::map<std::string, const Spectrum *> &_spectra,
    std::map<std::string, const Material *> _materials,
    const std::map<std::string, const FloatTexture *> &_float_textures,
    const std::map<std::string, const SpectrumTexture *> &_albedo_spectrum_textures,
    const std::map<std::string, const SpectrumTexture *> &_illuminant_spectrum_textures,
    const std::map<std::string, const SpectrumTexture *> &_unbounded_spectrum_textures,
    std::vector<void *> &gpu_dynamic_pointers)
    : root(_root), global_spectra(_global_spectra), spectra(_spectra), materials(_materials),
      float_textures(_float_textures), albedo_spectrum_textures(_albedo_spectrum_textures),
      illuminant_spectrum_textures(_illuminant_spectrum_textures),
      unbounded_spectrum_textures(_unbounded_spectrum_textures) {
    // the 1st token is Keyword
    // the 2nd token is String
    // e.g. { Shape "trianglemesh" }, { Camera "perspective" }

    for (int idx = 0; idx < tokens.size(); idx += 2) {
        if (tokens[idx].type != TokenType::Variable) {
            std::cout << "tokens[" << idx << "] is not a Variable\n";
            std::cout << "tokens[" << idx << "]: " << tokens[idx] << "\n";

            REPORT_FATAL_ERROR();
        }

        auto variable_type = tokens[idx].values[0];
        auto variable_name = tokens[idx].values[1];

        if (variable_type == "blackbody") {
            auto value_in_str = tokens[idx + 1].values[0];
            auto variable_value = std::stod(value_in_str);

            blackbodies[variable_name] = variable_value;
            continue;
        }

        if (variable_type == "bool") {
            auto value_in_str = tokens[idx + 1].values[0];

            if (value_in_str == "true") {
                booleans[variable_name] = true;
                continue;
            }

            if (value_in_str == "false") {
                booleans[variable_name] = false;
                continue;
            }

            printf("\n%s(): illegal BOOL value: %s\n", __func__, value_in_str.c_str());
            REPORT_FATAL_ERROR();
        }

        if (variable_type == "float") {
            floats[variable_name] = tokens[idx + 1].to_floats();
            continue;
        }

        if (variable_type == "integer") {
            integers[variable_name] = tokens[idx + 1].to_integers();
            continue;
        }

        if (variable_type == "normal") {
            auto numbers = tokens[idx + 1].to_floats();
            auto n = std::vector<Normal3f>(numbers.size() / 3);
            for (int k = 0; k < n.size(); k++) {
                n[k] = Normal3f(numbers[k * 3], numbers[k * 3 + 1], numbers[k * 3 + 2]);
            }
            normals[variable_name] = n;
            continue;
        }

        if (variable_type == "point2") {
            auto numbers = tokens[idx + 1].to_floats();
            auto p = std::vector<Point2f>(numbers.size() / 2);
            for (int k = 0; k < p.size(); k++) {
                p[k] = Point2f(numbers[k * 2], numbers[k * 2 + 1]);
            }

            point2s[variable_name] = p;
            continue;
        }

        if (variable_type == "point3") {
            auto numbers = tokens[idx + 1].to_floats();
            auto p = std::vector<Point3f>(numbers.size() / 3);
            for (int k = 0; k < p.size(); k++) {
                p[k] = Point3f(numbers[k * 3], numbers[k * 3 + 1], numbers[k * 3 + 2]);
            }

            point3s[variable_name] = p;
            continue;
        }

        if (variable_type == "rgb") {
            auto rgb_list = tokens[idx + 1].to_floats();
            rgbs[variable_name] = RGB(rgb_list[0], rgb_list[1], rgb_list[2]);
            continue;
        }

        if (variable_type == "spectrum") {
            auto spectrum_arg_list = tokens[idx + 1].values;

            if (spectrum_arg_list.empty()) {
                REPORT_FATAL_ERROR();
            }

            if (spectrum_arg_list.size() == 1) {
                // the only value could be a path
                auto spectrum_arg = spectrum_arg_list[0];

                auto file_path = root + "/" + spectrum_arg;
                if (std::filesystem::is_regular_file(file_path)) {
                    auto spectrum_samples = read_spectrum_file(file_path);

                    auto built_spectrum =
                        Spectrum::create_piecewise_linear_spectrum_from_interleaved(
                            std::vector(std::begin(spectrum_samples), std::end(spectrum_samples)),
                            false, nullptr, gpu_dynamic_pointers);

                    spectra[variable_name] = built_spectrum;
                    continue;
                }

                if (_spectra.find(spectrum_arg) != _spectra.end()) {
                    // or name of a spectrum
                    spectra[variable_name] = _spectra.at(spectrum_arg);
                    continue;
                }
            }

            // build PiecewiseLinearSpectrum from Interleaved data
            std::vector<FloatType> floats;
            for (const auto &x : spectrum_arg_list) {
                floats.push_back(stod(x));
            }

            auto spectrum = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
                floats, false, nullptr, gpu_dynamic_pointers);

            spectra[variable_name] = spectrum;
            continue;
        }

        if (variable_type == "string") {
            strings[variable_name] = tokens[idx + 1].values;
            continue;
        }

        if (variable_type == "texture") {
            textures_name[variable_name] = tokens[idx + 1].values[0];
            continue;
        }

        if (variable_type == "vector3") {
            auto value_list = tokens[idx + 1].to_floats();
            vector3s[variable_name] = Vector3f(value_list[0], value_list[1], value_list[2]);
            continue;
        }

        printf("\n%s(): unknown variable type: %s\n", __func__, variable_type.c_str());
        REPORT_FATAL_ERROR();
    }
}

const Spectrum *ParameterDictionary::get_spectrum(const std::string &key,
                                                  SpectrumType spectrum_type,
                                                  std::vector<void *> &gpu_dynamic_pointers) const {
    if (spectra.find(key) != spectra.end()) {
        return spectra.at(key);
    }

    if (has_rgb(key)) {
        auto rgb_val = rgbs.at(key);
        return Spectrum::create_from_rgb(rgb_val, spectrum_type, global_spectra->rgb_color_space,
                                         gpu_dynamic_pointers);
    }

    if (blackbodies.find(key) != blackbodies.end()) {
        auto value = blackbodies.at(key);
        return Spectrum::create_black_body(value, gpu_dynamic_pointers);
    }

    if (DEBUG_MODE) {
        printf("%s(): key `%s` not found in Spectrum, RGB, Blackbody\n", __func__, key.c_str());
    }

    return nullptr;
}

const Material *ParameterDictionary::get_material(const std::string &key) const {
    return materials.at(key);
}

const FloatTexture *
ParameterDictionary::get_float_texture_or_null(const std::string &key,
                                               std::vector<void *> &gpu_dynamic_pointers) const {
    if (textures_name.find(key) != textures_name.end()) {
        const auto tex_name = textures_name.at(key);

        if (float_textures.find(tex_name) != float_textures.end()) {
            return float_textures.at(tex_name);
        }

        return nullptr;
    }

    if (has_floats(key)) {
        auto val = get_float(key);
        return FloatTexture::create_constant_float_texture(val, gpu_dynamic_pointers);
    }

    if (DEBUG_MODE) {
        printf("`%s` not found in FloatTexture\n", key.c_str());
    }

    return nullptr;
}

const FloatTexture *
ParameterDictionary::get_float_texture(const std::string &key, FloatType default_val,
                                       std::vector<void *> &gpu_dynamic_pointers) const {
    auto texture = get_float_texture_or_null(key, gpu_dynamic_pointers);
    if (texture) {
        return texture;
    }

    return FloatTexture::create_constant_float_texture(default_val, gpu_dynamic_pointers);
}

const FloatTexture *ParameterDictionary::get_float_texture_with_default_val(
    const std::string &key, FloatType default_val,
    std::vector<void *> &gpu_dynamic_pointers) const {
    auto texture = get_float_texture_or_null(key, gpu_dynamic_pointers);
    if (texture) {
        return texture;
    }

    return FloatTexture::create_constant_float_texture(default_val, gpu_dynamic_pointers);
}

const SpectrumTexture *
ParameterDictionary::get_spectrum_texture(const std::string &key, SpectrumType spectrum_type,
                                          std::vector<void *> &gpu_dynamic_pointers) const {
    switch (spectrum_type) {
    case (SpectrumType::Albedo):
    case (SpectrumType::Illuminant):
    case (SpectrumType::Unbounded): {
        break;
    }
    default: {
        REPORT_FATAL_ERROR();
    }
    }

    if (textures_name.find(key) != textures_name.end()) {
        auto tex_name = textures_name.at(key);

        auto &spectrumTextures =
            spectrum_type == SpectrumType::Albedo
                ? albedo_spectrum_textures
                : (spectrum_type == SpectrumType::Illuminant ? illuminant_spectrum_textures
                                                             : unbounded_spectrum_textures);

        if (spectrumTextures.find(tex_name) != spectrumTextures.end()) {
            return spectrumTextures.at(tex_name);
        }

        printf("ERROR: spectrum texture not found: `%s` -> `%s`\n", key.c_str(), tex_name.c_str());

        REPORT_FATAL_ERROR();

        return nullptr;
    }

    auto spectrum = get_spectrum(key, spectrum_type, gpu_dynamic_pointers);
    if (spectrum) {
        return SpectrumTexture::create_constant_texture(spectrum, gpu_dynamic_pointers);
    }

    if (DEBUG_MODE) {
        printf("WARNING: spectrum texture not found: `%s`\n", key.c_str());
    }

    return nullptr;
}
