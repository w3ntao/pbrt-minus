#pragma once

#include <map>
#include <optional>
#include <vector>

#include "pbrt/spectrum_util/rgb.h"
#include "pbrt/scene/tokenizer.h"

class SpectrumTexture;
class FloatTexture;

class ParameterDict {
  public:
    ParameterDict() = default;

    explicit ParameterDict(
        const std::vector<Token> &tokens,
        const std::map<std::string, const SpectrumTexture *> &named_spectrum_texture,
        const std::string &_root)
        : root(_root) {
        // the 1st token is Keyword
        // the 2nd token is String
        // e.g. { Shape "trianglemesh" }, { Camera "perspective" }

        for (int idx = 0; idx < tokens.size(); idx += 2) {
            if (tokens[idx].type != TokenType::Variable) {
                std::cout << "tokens[" << idx << "] is not a Variable\n\n";
                throw std::runtime_error("expect token Variable");
            }

            auto variable_type = tokens[idx].values[0];
            auto variable_name = tokens[idx].values[1];

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

            if (variable_type == "string") {
                strings[variable_name] = tokens[idx + 1].values[0];
                continue;
            }

            if (variable_type == "texture") {
                auto target_texture_name = tokens[idx + 1].values[0];

                if (named_spectrum_texture.find(target_texture_name) ==
                    named_spectrum_texture.end()) {
                    printf("texture not found in spectrum texture: %s\n\n",
                           target_texture_name.c_str());
                    REPORT_FATAL_ERROR();
                }

                spectrum_textures[variable_name] = named_spectrum_texture.at(target_texture_name);
                continue;

                REPORT_FATAL_ERROR();
            }

            printf("\n%s(): unknown variable type: %s\n", __func__, variable_type.c_str());
            REPORT_FATAL_ERROR();
        }
    }

    bool has_floats(const std::string &key) const {
        return floats.find(key) != floats.end();
    }

    bool has_string(const std::string &key) const {
        return strings.find(key) != strings.end();
    }

    bool has_rgb(const std::string &key) const {
        return rgbs.find(key) != rgbs.end();
    }

    bool has_float_texture(const std::string &key) const {
        return float_textures.find(key) != float_textures.end();
    }

    bool has_spectrum_texture(const std::string &key) const {
        return spectrum_textures.find(key) != spectrum_textures.end();
    }

    std::vector<int> get_integer(const std::string &key) const {
        if (integers.find(key) == integers.end()) {
            return {};
        }

        return integers.at(key);
    }

    FloatType get_float(const std::string &key, std::optional<FloatType> default_val) const {
        if (floats.find(key) == floats.end()) {
            if (!default_val.has_value()) {
                printf("%s(): key not available\n", __func__);
                REPORT_FATAL_ERROR();
            }

            return default_val.value();
        }

        return floats.at(key)[0];
    }

    bool get_bool(const std::string &key, std::optional<bool> default_val) const {
        if (booleans.find(key) == booleans.end()) {
            if (!default_val.has_value()) {
                printf("%s(): key not available\n", __func__);
                REPORT_FATAL_ERROR();
            }

            return default_val.value();
        }

        return booleans.at(key);
    }

    std::string get_string(const std::string &key, std::optional<std::string> default_val) const {
        if (strings.find(key) == strings.end()) {
            if (!default_val.has_value()) {
                printf("%s(): key not available\n", __func__);
                REPORT_FATAL_ERROR();
            }

            return default_val.value();
        }

        return strings.at(key);
    }

    std::vector<Point2f> get_point2(const std::string &key) const {
        if (point2s.find(key) == point2s.end()) {
            return {};
        }

        return point2s.at(key);
    }

    std::vector<Point3f> get_point3(const std::string &key) const {
        if (point3s.find(key) == point3s.end()) {
            return {};
        }

        return point3s.at(key);
    }

    RGB get_rgb(const std::string &key, std::optional<RGB> default_val) const {
        if (rgbs.find(key) == rgbs.end()) {
            if (!default_val.has_value()) {
                printf("%s(): key not available\n", __func__);
                REPORT_FATAL_ERROR();
            }

            return default_val.value();
        }

        return rgbs.at(key);
    }

    const FloatTexture *get_float_texture(const std::string &key) const {
        return float_textures.at(key);
    }

    const SpectrumTexture *get_spectrum_texture(const std::string &key) const {
        return spectrum_textures.at(key);
    }

    friend std::ostream &operator<<(std::ostream &stream, const ParameterDict &parameters) {
        if (!parameters.integers.empty()) {
            stream << "integers:\n";
            parameters.print_dict_of_vector(stream, parameters.integers);
            stream << "\n";
        }

        if (!parameters.point2s.empty()) {
            stream << "Poin2f:\n";
            parameters.print_dict_of_vector(stream, parameters.point2s);
            stream << "\n";
        }

        if (!parameters.point3s.empty()) {
            stream << "Poin3f:\n";
            parameters.print_dict_of_vector(stream, parameters.point3s);
            stream << "\n";
        }

        if (!parameters.normals.empty()) {
            stream << "Normal3f:\n";
            parameters.print_dict_of_vector(stream, parameters.normals);
            stream << "\n";
        }

        if (!parameters.floats.empty()) {
            stream << "Float:\n";
            parameters.print_dict_of_vector(stream, parameters.floats);
            stream << "\n";
        }

        if (!parameters.strings.empty()) {
            stream << "String:\n";
            parameters.print_dict_of_single_var(stream, parameters.strings);
            stream << "\n";
        }

        if (!parameters.rgbs.empty()) {
            stream << "RGB:\n";
            parameters.print_dict_of_single_var(stream, parameters.rgbs);
            stream << "\n";
        }

        if (!parameters.booleans.empty()) {
            stream << "bool:\n";
            parameters.print_dict_of_single_var(stream, parameters.booleans);
            stream << "\n";
        }

        return stream;
    }

    std::string root;

  private:
    std::map<std::string, bool> booleans;
    std::map<std::string, std::vector<Point2f>> point2s;
    std::map<std::string, std::vector<Point3f>> point3s;
    std::map<std::string, std::vector<Normal3f>> normals;
    std::map<std::string, std::vector<int>> integers;
    std::map<std::string, std::vector<FloatType>> floats;
    std::map<std::string, std::string> strings;
    std::map<std::string, RGB> rgbs;
    std::map<std::string, const FloatTexture *> float_textures;
    std::map<std::string, const SpectrumTexture *> spectrum_textures;

    template <typename T>
    void print_dict_of_single_var(std::ostream &stream,
                                  const std::map<std::string, T> kv_map) const {
        for (const auto &kv : kv_map) {
            stream << std::setprecision(4) << kv.first << ": " << kv.second << "\n";
        }
    }

    template <typename T>
    void print_dict_of_vector(std::ostream &stream,
                              const std::map<std::string, std::vector<T>> kv_map) const {
        for (const auto &kv : kv_map) {
            stream << kv.first << ": { ";
            for (const auto &x : kv.second) {
                stream << x << ", ";
            }
            stream << "}\n";
        }
    }
};
