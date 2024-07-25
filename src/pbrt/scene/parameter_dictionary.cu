#include "pbrt/scene/parameter_dictionary.h"

#include "pbrt/base/spectrum.h"
#include "pbrt/scene/tokenizer.h"
#include "pbrt/spectra/rgb_illuminant_spectrum.h"
#include "pbrt/spectrum_util/global_spectra.h"
#include "pbrt/spectra/black_body_spectrum.h"

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
    const std::vector<Token> &tokens, const std::map<std::string, const Spectrum *> &_named_spectra,
    const std::map<std::string, const FloatTexture *> &named_float_texture,
    const std::map<std::string, const SpectrumTexture *> &named_spectrum_texture,
    const std::string &_root, const GlobalSpectra *_global_spectra,
    std::vector<void *> &gpu_dynamic_pointers)
    : root(_root), global_spectra(_global_spectra), named_spectra(_named_spectra) {
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
            auto val_in_str = tokens[idx + 1].values[0];

            auto file_path = root + "/" + val_in_str;
            if (std::filesystem::is_regular_file(file_path)) {
                auto spectrum_samples = read_spectrum_file(file_path);

                auto built_spectrum = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
                    std::vector(std::begin(spectrum_samples), std::end(spectrum_samples)), false,
                    nullptr, gpu_dynamic_pointers);

                named_spectra[variable_name] = built_spectrum;
                continue;
            }

            if (_named_spectra.find(val_in_str) == _named_spectra.end()) {
                printf("\nERROR: spectrum `%s` not found\n", val_in_str.c_str());
                REPORT_FATAL_ERROR();
            }

            // otherwise it's a named spectrum
            named_spectra[variable_name] = _named_spectra.at(val_in_str);
            continue;
        }

        if (variable_type == "string") {
            strings[variable_name] = tokens[idx + 1].values[0];
            continue;
        }

        if (variable_type == "texture") {
            auto target_texture_name = tokens[idx + 1].values[0];

            if (named_spectrum_texture.find(target_texture_name) != named_spectrum_texture.end()) {
                spectrum_textures[variable_name] = named_spectrum_texture.at(target_texture_name);
                continue;
            }

            if (named_float_texture.find(target_texture_name) != named_float_texture.end()) {
                float_textures[variable_name] = named_float_texture.at(target_texture_name);
                continue;
            }

            printf("texture not found in texture: %s\n\n", target_texture_name.c_str());
            REPORT_FATAL_ERROR();
        }

        printf("\n%s(): unknown variable type: %s\n", __func__, variable_type.c_str());
        REPORT_FATAL_ERROR();
    }
}

const Spectrum *ParameterDictionary::get_spectrum(const std::string &key,
                                                  SpectrumType spectrum_type,
                                                  std::vector<void *> &gpu_dynamic_pointers) const {
    if (named_spectra.find(key) != named_spectra.end()) {
        return named_spectra.at(key);
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

    if (DEBUGGING) {
        printf("%s(): key `%s` not found in Spectrum, RGB, Blackbody\n", __func__, key.c_str());
    }
    
    return nullptr;
}
