#pragma once

#include "pbrt/base/light.h"
#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/transform.h"
#include "pbrt/scene/command_line_option.h"
#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/scene/parser.h"
#include "pbrt/util/thread_pool.h"
#include <filesystem>
#include <map>
#include <set>
#include <stack>

class BDPTIntegrator;
class Film;
class GlobalSpectra;
class Integrator;
class MLTPathIntegrator;
class Primitive;
class Renderer;
class WavefrontPathIntegrator;
struct IntegratorBase;

struct AreaLightEntity {
    std::string name;
    ParameterDictionary parameters;

    AreaLightEntity(const std::string &_name, const ParameterDictionary &_parameters)
        : name(_name), parameters(_parameters) {}
};

class GraphicsState {
  public:
    GraphicsState() : transform(Transform::identity()), reverse_orientation(false) {}

    bool reverse_orientation = false;
    Transform transform;
    const Material *material = nullptr;

    std::optional<AreaLightEntity> area_light_entity;
};

class SceneBuilder {
    ThreadPool thread_pool;
    std::string root;

    [[nodiscard]] std::string get_file_full_path(const std::string &relative_path) const {
        return root.empty() ? relative_path : root + "/" + relative_path;
    }

    static std::string get_dirname(const std::string &full_path) {
        const size_t last_slash_idx = full_path.rfind('/');
        if (std::string::npos != last_slash_idx) {
            return full_path.substr(0, last_slash_idx);
        }

        throw std::runtime_error("get_dirname() fails");
    }

    std::optional<int> samples_per_pixel;
    std::optional<std::string> integrator_name;
    bool preview = false;

    const Integrator *megakernel_integrator = nullptr;
    WavefrontPathIntegrator *wavefront_integrator = nullptr;
    BDPTIntegrator *bdpt_integrator = nullptr;
    MLTPathIntegrator *mlt_integrator = nullptr;

    IntegratorBase *integrator_base = nullptr;

    Film *film = nullptr;

    const GlobalSpectra *global_spectra = nullptr;

    std::vector<void *> gpu_dynamic_pointers;

    std::map<std::string, const Material *> materials;
    std::map<std::string, const Spectrum *> spectra;

    std::map<std::string, const FloatTexture *> float_textures;

    std::map<std::string, const SpectrumTexture *> albedo_spectrum_textures;
    std::map<std::string, const SpectrumTexture *> illuminant_spectrum_textures;
    std::map<std::string, const SpectrumTexture *> unbounded_spectrum_textures;

    std::string output_filename;

    std::vector<Token> camera_tokens;
    std::vector<Token> film_tokens;
    std::vector<Token> integrator_tokens;
    std::vector<Token> pixel_filter_tokens;

    std::vector<const Primitive *> gpu_primitives;
    std::vector<Light *> gpu_lights;

    GraphicsState graphics_state;
    std::stack<GraphicsState> pushed_graphics_state;
    std::map<std::string, Transform> named_coordinate_systems;
    Transform render_from_world;

    struct InstantiatedPrimitive {
        const Primitive *primitives = nullptr;
        uint num;

        InstantiatedPrimitive(const Primitive *_primitives, uint _num)
            : primitives(_primitives), num(_num) {}
    };

    struct ActiveInstanceDefinition {
        ActiveInstanceDefinition() = default;
        std::string name;
        std::vector<InstantiatedPrimitive> instantiated_primitives;
    };

    std::shared_ptr<ActiveInstanceDefinition> active_instance_definition = nullptr;

    std::map<std::string, std::shared_ptr<ActiveInstanceDefinition>> instance_definition;

  public:
    explicit SceneBuilder(const CommandLineOption &command_line_option);

    ~SceneBuilder() {
        for (auto ptr : gpu_dynamic_pointers) {
            CHECK_CUDA_ERROR(cudaFree(ptr));
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    ParameterDictionary build_parameter_dictionary(const std::vector<Token> &tokens) {
        return ParameterDictionary(tokens, root, global_spectra, spectra, materials, float_textures,
                                   albedo_spectrum_textures, illuminant_spectrum_textures,
                                   unbounded_spectrum_textures, gpu_dynamic_pointers);
    }

    void build_camera();

    void build_filter();

    void build_film();

    void build_gpu_lights();

    void build_integrator();

    void parse_keyword(const std::vector<Token> &tokens);

    void parse_area_light_source(const std::vector<Token> &tokens);

    void parse_concat_transform(const std::vector<Token> &tokens);

    void parse_light_source(const std::vector<Token> &tokens);

    void parse_lookat(const std::vector<Token> &tokens);

    void parse_make_named_material(const std::vector<Token> &tokens);

    void parse_material(const std::vector<Token> &tokens);

    void parse_named_material(const std::vector<Token> &tokens);

    void parse_rotate(const std::vector<Token> &tokens);

    void parse_scale(const std::vector<Token> &tokens);

    void parse_shape(const std::vector<Token> &tokens);

    void parse_texture(const std::vector<Token> &tokens);

    void parse_transform(const std::vector<Token> &tokens);

    void parse_translate(const std::vector<Token> &tokens);

    void parse_tokens(const std::vector<Token> &tokens);

    Transform get_render_from_object() const {
        return render_from_world * graphics_state.transform;
    }

    void parse_file(const std::string &_filename);

    void preprocess();

    void render() const;

    static void render_pbrt(const CommandLineOption &command_line_option) {
        if (!std::filesystem::is_regular_file(command_line_option.input_file)) {
            std::cout << "file not found: `" + command_line_option.input_file + "`\n\n";
            exit(1);
        }

        auto builder = SceneBuilder(command_line_option);

        auto input_file = command_line_option.input_file;

        if (std::filesystem::path p(input_file); p.extension() != ".pbrt") {
            printf("ERROR: input file `%s` not ended with `.pbrt`\n", input_file.c_str());
            REPORT_FATAL_ERROR();
        }

        builder.root = get_dirname(input_file);

        builder.parse_file(input_file);

        builder.preprocess();

        builder.render();
    }
};
