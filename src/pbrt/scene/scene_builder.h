#pragma once

#include <chrono>
#include <filesystem>
#include <map>
#include <set>
#include <stack>

#include "pbrt/base/light.h"

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/transform.h"

#include "pbrt/gpu/renderer.h"

#include "pbrt/scene/command_line_option.h"
#include "pbrt/scene/parser.h"
#include "pbrt/scene/parameter_dict.h"
#include "pbrt/scene/precomputed_spectrum.h"

#include "pbrt/util/std_container.h"
#include "pbrt/util/thread_pool.h"

struct AreaLightEntity {
    std::string name;
    ParameterDict parameters;

    AreaLightEntity(const std::string &_name, const ParameterDict &_parameters)
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
  private:
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

  public:
    std::optional<int> samples_per_pixel;
    std::optional<std::string> integrator_name;

    GPU::Renderer *renderer = nullptr;

    PreComputedSpectrum pre_computed_spectrum;
    std::vector<void *> gpu_dynamic_pointers;

    std::optional<Point2i> film_resolution = std::nullopt;
    std::string output_filename;
    std::vector<Token> camera_tokens;
    std::vector<Token> film_tokens;
    std::vector<Token> sampler_tokens;

    std::vector<const Primitive *> gpu_primitives;
    std::vector<Light *> gpu_lights;

    GraphicsState graphics_state;
    std::stack<GraphicsState> pushed_graphics_state;
    std::map<std::string, Transform> named_coordinate_systems;
    Transform render_from_world;

    std::map<std::string, const SpectrumTexture *> named_spectrum_texture;
    std::map<std::string, const Material *> named_material;

    explicit SceneBuilder(const CommandLineOption &command_line_option);

    ~SceneBuilder() {
        for (auto ptr : gpu_dynamic_pointers) {
            CHECK_CUDA_ERROR(cudaFree(ptr));
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    void build_camera();

    void build_filter();

    void build_film();

    void build_sampler();

    void build_integrator();

    const Material *create_material(const std::string &type_of_material,
                                    const ParameterDict &parameters);

    void parse_area_light_source(const std::vector<Token> &tokens);

    void parse_light_source(const std::vector<Token> &tokens);

    void parse_lookat(const std::vector<Token> &tokens) {
        if (tokens[0] != Token(TokenType::Keyword, "LookAt")) {
            throw std::runtime_error("expect Keyword(LookAt)");
        }

        std::vector<FloatType> data;
        for (int idx = 1; idx < tokens.size(); idx++) {
            data.push_back(tokens[idx].to_float());
        }

        auto position = Point3f(data[0], data[1], data[2]);
        auto look = Point3f(data[3], data[4], data[5]);
        auto up = Vector3f(data[6], data[7], data[8]);

        graphics_state.transform = graphics_state.transform * Transform::lookat(position, look, up);
    }

    void parse_make_named_material(const std::vector<Token> &tokens);

    void parse_material(const std::vector<Token> &tokens);

    void parse_named_material(const std::vector<Token> &tokens);

    void parse_rotate(const std::vector<Token> &tokens);

    void parse_scale(const std::vector<Token> &tokens);

    void parse_shape(const std::vector<Token> &tokens);

    void parse_texture(const std::vector<Token> &tokens);

    void parse_transform(const std::vector<Token> &tokens);

    void parse_translate(const std::vector<Token> &tokens);

    void add_triangle_mesh(const std::vector<Point3f> &points, const std::vector<int> &indices,
                           const std::vector<Point2f> &uv);

    void parse_tokens(const std::vector<Token> &tokens);

    Transform get_render_from_object() const {
        return render_from_world * graphics_state.transform;
    }

    void parse_file(const std::string &_filename);

    void preprocess();

    void render() const;

  public:
    static void render_pbrt(const CommandLineOption &command_line_option) {
        if (!std::filesystem::exists(command_line_option.input_file)) {
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
