#pragma once

#include <stack>
#include <map>
#include <chrono>

#include "pbrt/scene/command_line_option.h"
#include "pbrt/scene/parser.h"
#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/transform.h"
#include "pbrt/scene/parameter_dict.h"
#include "pbrt/gpu/rendering.cuh"

class GraphicsState {
  public:
    GraphicsState() : current_transform(Transform::identity()), reverse_orientation(false) {}

    Transform current_transform;
    bool reverse_orientation;
};

class SceneBuilder {
  private:
    std::optional<int> samples_per_pixel;
    std::optional<std::string> integrator;

    GPU::Renderer *renderer = nullptr;
    std::optional<Point2i> resolution = std::nullopt;
    std::string filename;
    std::vector<Token> lookat_tokens;
    std::vector<Token> camera_tokens;
    std::vector<Token> film_tokens;
    std::vector<Token> sampler_tokens;

    GraphicsState graphics_state;
    std::stack<GraphicsState> pushed_graphics_state;
    std::map<std::string, Transform> named_coordinate_systems;
    Transform render_from_world;

    SceneBuilder(const CommandLineOption &command_line_option)
        : samples_per_pixel(command_line_option.samples_per_pixel),
          integrator(command_line_option.integrator) {}

    std::vector<int> group_tokens(const std::vector<Token> &tokens) {
        std::vector<int> keyword_range;
        for (int idx = 0; idx < tokens.size(); ++idx) {
            const auto &token = tokens[idx];
            if (token.type == WorldBegin || token.type == AttributeBegin ||
                token.type == AttributeEnd || token.type == Keyword) {
                keyword_range.push_back(idx);
            }
        }
        keyword_range.push_back(tokens.size());

        return keyword_range;
    }

    Transform get_render_from_object() const {
        return render_from_world * graphics_state.current_transform;
    }

    void option_camera() {
        auto parameters = ParameterDict(camera_tokens);
        const auto camera_type = camera_tokens[1].value[0];
        if (camera_type == "perspective") {
            auto camera_from_world = graphics_state.current_transform;
            auto world_from_camera = camera_from_world.inverse();

            named_coordinate_systems["camera"] = world_from_camera;

            auto camera_transform = CameraTransform(
                world_from_camera, RenderingCoordinateSystem::CameraWorldCoordSystem);

            render_from_world = camera_transform.render_from_world;

            double fov = 90;
            if (const auto _fov = parameters.get_float("fov"); _fov.size() > 0) {
                fov = _fov[0];
            }

            gpu_init_camera<<<1, 1>>>(renderer, resolution.value(), camera_transform, fov);
            return;
        }

        std::cerr << "Camera type `" << camera_type << "` not implemented\n";
        throw std::runtime_error("camera type not implemented");
    }

    void option_film() {
        auto parameters = ParameterDict(film_tokens);

        auto _resolution_x = parameters.get_integer("xresolution")[0];
        auto _resolution_y = parameters.get_integer("yresolution")[0];

        resolution = Point2i(_resolution_x, _resolution_y);
        filename = parameters.get_string("filename");
    }

    void option_lookat() {
        std::vector<double> data;
        for (int idx = 1; idx < lookat_tokens.size(); idx++) {
            data.push_back(lookat_tokens[idx].to_number());
        }

        auto position = Point3f(data[0], data[1], data[2]);
        auto look = Point3f(data[3], data[4], data[5]);
        auto up = Vector3f(data[6], data[7], data[8]);

        auto transform_look_at = Transform::lookat(position, look, up);

        graphics_state.current_transform *= transform_look_at;
    }

    void option_sampler() {
        // TODO: sampler is not parsed, only pixelsamples read
        const auto parameters = ParameterDict(sampler_tokens);
        auto samples_from_parameters = parameters.get_integer("pixelsamples");

        if (!samples_per_pixel) {
            if (!samples_from_parameters.empty()) {
                samples_per_pixel = samples_from_parameters[0];
            } else {
                samples_per_pixel = 4;
                // default samples per pixel
            }
        }
    }

    void world_shape(const std::vector<Token> &tokens) {
        if (tokens[0] != Token(Keyword, "Shape")) {
            throw std::runtime_error("expect Keyword(Shape)");
        }

        auto render_from_object = get_render_from_object();
        bool reverse_orientation = graphics_state.reverse_orientation;

        auto second_token = tokens[1];

        const auto parameters = ParameterDict(tokens);

        if (second_token.value[0] == "trianglemesh") {
            auto uv = parameters.get_point2("uv");
            auto indices = parameters.get_integer("indices");
            auto points = parameters.get_point3("P");

            Point3f *gpu_points;
            checkCudaErrors(
                cudaMallocManaged((void **)&gpu_points, sizeof(Point3f) * points.size()));
            checkCudaErrors(cudaMemcpy(gpu_points, points.data(), sizeof(Point3f) * points.size(),
                                       cudaMemcpyHostToDevice));

            int *gpu_indicies;
            checkCudaErrors(
                cudaMallocManaged((void **)&gpu_indicies, sizeof(int) * indices.size()));
            checkCudaErrors(cudaMemcpy(gpu_indicies, indices.data(), sizeof(int) * indices.size(),
                                       cudaMemcpyHostToDevice));

            Point2f *gpu_uv;
            checkCudaErrors(cudaMallocManaged((void **)&gpu_uv, sizeof(Point2f) * uv.size()));
            checkCudaErrors(
                cudaMemcpy(gpu_uv, uv.data(), sizeof(Point2f) * uv.size(), cudaMemcpyHostToDevice));

            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            gpu_add_triangle_mesh<<<1, 1>>>(renderer, render_from_object, reverse_orientation,
                                            gpu_points, points.size(), gpu_indicies, indices.size(),
                                            gpu_uv, uv.size());

            checkCudaErrors(cudaFree(gpu_points));
            checkCudaErrors(cudaFree(gpu_indicies));
            checkCudaErrors(cudaFree(gpu_uv));

            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }

    void world_translate(const std::vector<Token> &tokens) {
        std::vector<double> data;
        for (int idx = 1; idx < tokens.size(); idx++) {
            data.push_back(tokens[idx].to_number());
        }

        graphics_state.current_transform *= Transform::translate(data[0], data[1], data[2]);
    }

    void parse_tokens(const std::vector<Token> &tokens) {
        const Token &first_token = tokens[0];

        switch (first_token.type) {
        case AttributeBegin: {
            pushed_graphics_state.push(graphics_state);
            return;
        }
        case AttributeEnd: {
            graphics_state = pushed_graphics_state.top();
            pushed_graphics_state.pop();
            return;
        }
        case WorldBegin: {
            checkCudaErrors(cudaMallocManaged((void **)&renderer, sizeof(GPU::Renderer)));
            gpu_init_renderer<<<1, 1>>>(renderer);

            option_lookat();
            option_film();
            option_camera();
            option_sampler();

            gpu_init_integrator<<<1, 1>>>(renderer);

            graphics_state.current_transform = Transform::identity();
            named_coordinate_systems["world"] = graphics_state.current_transform;

            return;
        }
        case Keyword: {
            const auto keyword = first_token.value[0];

            if (keyword == "Camera") {
                camera_tokens = tokens;
                return;
            }

            if (keyword == "Film") {
                film_tokens = tokens;
                return;
            }

            if (keyword == "LookAt") {
                lookat_tokens = tokens;
                return;
            }

            if (keyword == "Sampler") {
                sampler_tokens = tokens;
                return;
            }

            if (keyword == "Shape") {
                world_shape(tokens);
                return;
            }

            if (keyword == "Translate") {
                world_translate(tokens);
                return;
            }

            std::cout << "parse_tokens::Keyword `" << keyword << "` not implemented\n";

            return;
        }

        default: {
            printf("unkown token type: `%d`\n", first_token.type);
            throw std::runtime_error("parse_tokens() fail");
        }
        }
    }

  public:
    ~SceneBuilder() {
        if (renderer == nullptr) {
            return;
        }

        gpu_free_renderer<<<1, 1>>>(renderer);
        checkCudaErrors(cudaFree(renderer));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        cudaDeviceReset();

        renderer = nullptr;
    }

    static void render_pbrt(const CommandLineOption &command_line_option) {
        if (!std::filesystem::exists(command_line_option.input_file)) {
            std::cout << "file not found: `" + command_line_option.input_file + "`\n\n";
            exit(1);
        }

        const auto all_tokens = parse_pbrt_into_token(command_line_option.input_file);

        auto builder = SceneBuilder(command_line_option);
        const auto range_of_tokens = builder.group_tokens(all_tokens);

        for (int range_idx = 0; range_idx < range_of_tokens.size() - 1; ++range_idx) {
            auto current_tokens = std::vector(all_tokens.begin() + range_of_tokens[range_idx],
                                              all_tokens.begin() + range_of_tokens[range_idx + 1]);

            builder.parse_tokens(current_tokens);
        }

        builder.render();
    }

    void render() const {
        int thread_width = 8;
        int thread_height = 8;

        std::cout << "\n";
        std::cout << "rendering a " << resolution->x << "x" << resolution->y
                  << " image (samples per pixel: " << samples_per_pixel.value() << ") ";
        std::cout << "in " << thread_width << "x" << thread_height << " blocks.\n";

        gpu_aggregate_preprocess<<<1, 1>>>(renderer);

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // allocate FB
        RGB *frame_buffer;
        checkCudaErrors(
            cudaMallocManaged((void **)&frame_buffer, sizeof(RGB) * resolution->x * resolution->y));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        auto start = std::chrono::system_clock::now();

        dim3 blocks(resolution->x / thread_width + 1, resolution->y / thread_height + 1, 1);
        dim3 threads(thread_width, thread_height, 1);

        gpu_parallel_render<<<blocks, threads>>>(frame_buffer, samples_per_pixel.value(), renderer);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        const std::chrono::duration<double> duration{std::chrono::system_clock::now() - start};

        std::cout << std::fixed << std::setprecision(1) << "took " << duration.count()
                  << " seconds.\n";

        GPU::writer_to_file(filename, frame_buffer, resolution->x, resolution->y);

        checkCudaErrors(cudaFree(frame_buffer));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        std::cout << "image saved to `" << filename << "`\n";
    }
};
