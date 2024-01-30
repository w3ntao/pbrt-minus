#pragma once

#include <stack>
#include <map>

#include "pbrt/scene/parser.h"
#include "pbrt/base/gpu_rendering.cuh"
#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/transform.h"

class GraphicsState {
  public:
    GraphicsState() : current_transform(Transform::identity()), reverse_orientation(false) {}

    Transform current_transform;
    bool reverse_orientation;
};

class ParameterDict {
  public:
    ParameterDict(const std::vector<Token> &tokens) {
        // the 1st token is Keyword
        // the 2nd token is String
        // e.g. { Shape "trianglemesh" }, { Camera "perspective" }

        for (int idx = 2; idx < tokens.size(); idx += 2) {
            if (tokens[idx].type != Variable) {
                throw std::runtime_error("expect token Variable");
            }

            auto variable_type = tokens[idx].value[0];
            auto variable_name = tokens[idx].value[1];

            if (variable_type == "integer") {
                integers[variable_name] = tokens[idx + 1].to_integers();
                continue;
            }

            if (variable_type == "float") {
                floats[variable_name] = tokens[idx + 1].to_floats();
                continue;
            }

            if (variable_type == "string") {
                strings[variable_name] = tokens[idx + 1].value[0];
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

            std::cout << "unkonwn variable type: `" << variable_type << "`\n";
            throw std::runtime_error("unkonwn variable type");
        }
    }

    std::vector<int> get_integer(const std::string &key) const {
        if (integers.find(key) == integers.end()) {
            return {};
        }

        return integers.at(key);
    }

    std::vector<double> get_float(const std::string &key) const {
        if (floats.find(key) == floats.end()) {
            return {};
        }

        return floats.at(key);
    }

    std::string get_string(const std::string &key) const {
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

    friend std::ostream &operator<<(std::ostream &stream, const ParameterDict &parameters) {
        if (!parameters.integers.empty()) {
            stream << "integers:\n";
            parameters.print_dict(stream, parameters.integers);
            stream << "\n";
        }

        if (!parameters.point2s.empty()) {
            stream << "Poin2f:\n";
            parameters.print_dict(stream, parameters.point2s);
            stream << "\n";
        }

        if (!parameters.point3s.empty()) {
            stream << "Poin3f:\n";
            parameters.print_dict(stream, parameters.point3s);
            stream << "\n";
        }

        return stream;
    }

  private:
    std::map<std::string, std::vector<Point2f>> point2s;
    std::map<std::string, std::vector<Point3f>> point3s;
    std::map<std::string, std::vector<int>> integers;
    std::map<std::string, std::vector<double>> floats;
    std::map<std::string, std::string> strings;

    template <typename T>
    void print_dict(std::ostream &stream,
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

class SceneBuilder {
  private:
    GPU::Renderer *renderer = nullptr;
    std::optional<Point2i> resolution = std::nullopt;
    std::string filename;
    std::vector<Token> lookat_tokens;
    std::vector<Token> camera_tokens;
    std::vector<Token> film_tokens;

    GraphicsState graphics_state;
    std::stack<GraphicsState> pushed_graphics_state;
    std::map<std::string, Transform> named_coordinate_systems;
    Transform render_from_world;

    SceneBuilder() {}

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

    void world_shape(const std::vector<Token> &tokens) {
        if (tokens[0] != Token(Keyword, "Shape")) {
            throw std::runtime_error("expect Keyword(Shape)");
        }

        auto render_from_object = get_render_from_object();
        bool reverse_orientation = graphics_state.reverse_orientation;

        auto second_token = tokens[1];

        if (second_token.value[0] == "trianglemesh") {
            const auto parameters = ParameterDict(tokens);

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

    static void render_pbrt(const std::string &filename) {
        const auto all_tokens = parse_pbrt_into_token(filename);

        auto builder = SceneBuilder();
        const auto range_of_tokens = builder.group_tokens(all_tokens);

        for (int range_idx = 0; range_idx < range_of_tokens.size() - 1; ++range_idx) {
            auto current_tokens = std::vector(all_tokens.begin() + range_of_tokens[range_idx],
                                              all_tokens.begin() + range_of_tokens[range_idx + 1]);

            builder.parse_tokens(current_tokens);
        }

        builder.render(1);
    }

    void render(int num_samples) const {
        int thread_width = 8;
        int thread_height = 8;

        const auto image_resolution = resolution.value();

        std::cerr << "Rendering a " << image_resolution.x << "x" << image_resolution.y
                  << " image (samples per pixel: " << num_samples << ") ";
        std::cerr << "in " << thread_width << "x" << thread_height << " blocks.\n";

        gpu_aggregate_preprocess<<<1, 1>>>(renderer);

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // allocate FB
        RGB *frame_buffer;
        checkCudaErrors(cudaMallocManaged((void **)&frame_buffer,
                                          sizeof(RGB) * image_resolution.x * image_resolution.y));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        clock_t start = clock();
        dim3 blocks(image_resolution.x / thread_width + 1, image_resolution.y / thread_height + 1,
                    1);
        dim3 threads(thread_width, thread_height, 1);

        gpu_parallel_render<<<blocks, threads>>>(frame_buffer, num_samples, renderer);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        const double timer_seconds = (double)(clock() - start) / CLOCKS_PER_SEC;
        std::cerr << std::fixed << std::setprecision(1) << "took " << timer_seconds
                  << " seconds.\n";

        GPU::writer_to_file(filename, frame_buffer, image_resolution.x, image_resolution.y);

        checkCudaErrors(cudaFree(frame_buffer));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        std::cout << "image saved to `" << filename << "`\n";
    }
};