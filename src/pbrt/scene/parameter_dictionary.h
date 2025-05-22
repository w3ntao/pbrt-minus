#pragma once

#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <pbrt/base/spectrum.h>
#include <pbrt/euclidean_space/normal3f.h>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/euclidean_space/point3.h>
#include <pbrt/spectrum_util/rgb.h>
#include <vector>

class FloatTexture;
class GPUMemoryAllocator;
class Material;
class Spectrum;
class SpectrumTexture;
class Token;
struct GlobalSpectra;

class ParameterDictionary {
  public:
    ParameterDictionary() = default;

    explicit ParameterDictionary(
        const std::vector<Token> &tokens, const std::string &_root,
        const GlobalSpectra *_global_spectra,
        const std::map<std::string, const Spectrum *> &_spectra,
        std::map<std::string, const Material *> _materials,
        const std::map<std::string, const FloatTexture *> &_float_textures,
        const std::map<std::string, const SpectrumTexture *> &_albedo_spectrum_textures,
        const std::map<std::string, const SpectrumTexture *> &_illuminant_spectrum_textures,
        const std::map<std::string, const SpectrumTexture *> &_unbounded_spectrum_textures,
        GPUMemoryAllocator &allocator);

    const GlobalSpectra *global_spectra = nullptr;

    bool has_floats(const std::string &key) const {
        return floats.find(key) != floats.end();
    }

    bool has_string(const std::string &key) const {
        return strings.find(key) != strings.end();
    }

    bool has_rgb(const std::string &key) const {
        return rgbs.find(key) != rgbs.end();
    }

    int get_integer(const std::string &key, std::optional<int> default_val = std::nullopt) const {
        if (integers.find(key) == integers.end()) {
            if (!default_val.has_value()) {
                printf("key `%s` not available\n", key.c_str());
                REPORT_FATAL_ERROR();
            }

            return default_val.value();
        }

        auto query_result = integers.at(key);
        if (query_result.size() > 1) {
            printf("key `%s` matched with more than 1 result\n", key.c_str());
            REPORT_FATAL_ERROR();
        }

        return query_result.at(0);
    }

    std::vector<int> get_integers(const std::string &key) const {
        if (integers.find(key) == integers.end()) {
            return {};
        }

        return integers.at(key);
    }

    Real get_float(const std::string &key,
                   const std::optional<Real> default_val = std::nullopt) const {
        if (floats.find(key) == floats.end()) {
            if (default_val.has_value()) {
                return default_val.value();
            }

            REPORT_FATAL_ERROR();
        }

        return floats.at(key)[0];
    }

    bool get_bool(const std::string &key, std::optional<bool> default_val) const {
        if (booleans.find(key) == booleans.end()) {
            if (!default_val.has_value()) {
                printf("key `%s` not available\n", key.c_str());
                REPORT_FATAL_ERROR();
            }

            return default_val.value();
        }

        return booleans.at(key);
    }

    std::string get_one_string(const std::string &key,
                               std::optional<std::string> default_val = std::nullopt) const {
        if (strings.find(key) == strings.end()) {
            if (!default_val.has_value()) {
                printf("%s(): key `%s` not available\n", __func__, key.c_str());
                REPORT_FATAL_ERROR();
            }

            return default_val.value();
        }

        auto result = strings.at(key);
        if (result.size() > 1) {
            REPORT_FATAL_ERROR();
        }

        return result.at(0);
    }

    std::vector<std::string> get_strings(const std::string &key) const {
        if (strings.find(key) == strings.end()) {
            return {};
        }

        return strings.at(key);
    }

    std::vector<Point2f> get_point2_array(const std::string &key) const {
        if (point2s.find(key) == point2s.end()) {
            return {};
        }

        return point2s.at(key);
    }

    Point3f get_point3(const std::string &key, const std::optional<Point3f> default_val) const {
        if (point3s.find(key) == point3s.end()) {
            if (!default_val.has_value()) {
                printf("%s(): key not available: %s\n", __func__, key.c_str());
                REPORT_FATAL_ERROR();
            }

            return default_val.value();
        }

        auto val = point3s.at(key);
        if (val.size() > 1) {
            printf("%s(): key `%s` is with multiple matched value\n", __func__, key.c_str());
            REPORT_FATAL_ERROR();
        }

        return val[0];
    }

    std::vector<Point3f> get_point3_array(const std::string &key) const {
        if (point3s.find(key) == point3s.end()) {
            return {};
        }

        return point3s.at(key);
    }

    Vector3f get_vector3f(const std::string &key,
                          const std::optional<Vector3f> default_val = std::nullopt) const {
        if (vector3s.find(key) == vector3s.end()) {
            if (default_val.has_value()) {
                return default_val.value();
            }

            REPORT_FATAL_ERROR();
        }

        return vector3s.at(key);
    }

    std::vector<Normal3f> get_normal_array(const std::string &key) const {
        if (normals.find(key) == normals.end()) {
            return {};
        }

        return normals.at(key);
    }

    const Spectrum *get_spectrum(const std::string &key, SpectrumType spectrum_type,
                                 GPUMemoryAllocator &allocator) const;

    const Material *get_material(const std::string &key) const;

    const FloatTexture *get_float_texture_or_null(const std::string &key,
                                                  GPUMemoryAllocator &allocator) const;

    const FloatTexture *get_float_texture(const std::string &key, Real default_val,
                                          GPUMemoryAllocator &allocator) const;

    const FloatTexture *get_float_texture_with_default_val(const std::string &key, Real default_val,
                                                           GPUMemoryAllocator &allocator) const;

    const SpectrumTexture *get_spectrum_texture(const std::string &key, SpectrumType spectrum_type,
                                                GPUMemoryAllocator &allocator) const;

    friend std::ostream &operator<<(std::ostream &stream, const ParameterDictionary &parameters) {
        if (!parameters.integers.empty()) {
            stream << "integers:\n";
            parameters.print_dict_of_vector(stream, parameters.integers);
            stream << "\n";
        }

        if (!parameters.floats.empty()) {
            stream << "Float:\n";
            parameters.print_dict_of_vector(stream, parameters.floats);
            stream << "\n";
        }

        if (!parameters.point2s.empty()) {
            stream << "Point2f:\n";
            parameters.print_dict_of_vector(stream, parameters.point2s);
            stream << "\n";
        }

        if (!parameters.point3s.empty()) {
            stream << "Point3f:\n";
            parameters.print_dict_of_vector(stream, parameters.point3s);
            stream << "\n";
        }

        if (!parameters.normals.empty()) {
            stream << "Normal3f:\n";
            parameters.print_dict_of_vector(stream, parameters.normals);
            stream << "\n";
        }

        if (!parameters.vector3s.empty()) {
            stream << "Vector3f:\n";
            parameters.print_dict_of_single_var(stream, parameters.vector3s);
            stream << "\n";
        }

        if (!parameters.spectra.empty()) {
            stream << "spectrum:\n";
            parameters.print_dict_of_single_var(stream, parameters.spectra);
            stream << "\n";
        }

        if (!parameters.rgbs.empty()) {
            stream << "RGB:\n";
            parameters.print_dict_of_single_var(stream, parameters.rgbs);
            stream << "\n";
        }

        if (!parameters.strings.empty()) {
            stream << "String:\n";
            parameters.print_dict_of_vector(stream, parameters.strings);
            stream << "\n";
        }

        if (!parameters.booleans.empty()) {
            stream << "bool:\n";
            parameters.print_dict_of_single_var(stream, parameters.booleans);
            stream << "\n";
        }

        if (!parameters.blackbodies.empty()) {
            stream << "blackbody:\n";
            parameters.print_dict_of_single_var(stream, parameters.blackbodies);
            stream << "\n";
        }

        if (!parameters.textures_name.empty()) {
            stream << "float texture:\n";
            parameters.print_dict_of_single_var(stream, parameters.textures_name);
            stream << "\n";
        }

        if (!parameters.albedo_spectrum_textures.empty()) {
            stream << "albedo spectrum texture:\n";
            parameters.print_dict_of_single_var(stream, parameters.albedo_spectrum_textures);
            stream << "\n";
        }

        if (!parameters.illuminant_spectrum_textures.empty()) {
            stream << "illuminant spectrum texture:\n";
            parameters.print_dict_of_single_var(stream, parameters.illuminant_spectrum_textures);
            stream << "\n";
        }

        if (!parameters.unbounded_spectrum_textures.empty()) {
            stream << "unbounded spectrum texture:\n";
            parameters.print_dict_of_single_var(stream, parameters.unbounded_spectrum_textures);
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
    std::map<std::string, Vector3f> vector3s;
    std::map<std::string, std::vector<int>> integers;
    std::map<std::string, std::vector<Real>> floats;
    std::map<std::string, std::vector<std::string>> strings;
    std::map<std::string, RGB> rgbs;
    std::map<std::string, Real> blackbodies;

    std::map<std::string, const Spectrum *> spectra;
    std::map<std::string, std::string> textures_name;
    std::map<std::string, const Material *> materials;

    std::map<std::string, const FloatTexture *> float_textures;

    std::map<std::string, const SpectrumTexture *> unbounded_spectrum_textures;
    std::map<std::string, const SpectrumTexture *> albedo_spectrum_textures;
    std::map<std::string, const SpectrumTexture *> illuminant_spectrum_textures;

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
