#pragma once

#include <stack>
#include <map>

#include "pbrt/scene/parser.h"
#include "pbrt/base/gpu_rendering.cuh"
#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/transform.h"

class GraphicsState {
  public:
    GraphicsState() : current_transform(Transform::identity()), reversed_orientation(false) {}

    Transform current_transform;
    bool reversed_orientation;
};

class SceneBuilder {
  private:
    Renderer *renderer = nullptr;
    Point2i resolution;
    std::vector<Token> lookat_tokens;
    std::vector<Token> camera_tokens;

    GraphicsState graphics_state;
    std::stack<GraphicsState> pushed_graphics_state;
    std::map<std::string, Transform> named_coordinate_systems;
    Transform render_from_world;

    SceneBuilder() : resolution(-1, -1) {}

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
        if (camera_tokens[1].type != String) {
            throw std::runtime_error("option_camera() fail: the 2nd token should be String");
        }

        const auto camera_type = camera_tokens[1].value[0];
        if (camera_type == "perspective") {
            auto camera_from_world = graphics_state.current_transform;
            auto world_from_camera = camera_from_world.inverse();

            named_coordinate_systems["camera"] = world_from_camera;

            auto camera_transform = CameraTransform(
                world_from_camera, RenderingCoordinateSystem::CameraWorldCoordSystem);

            render_from_world = camera_transform.render_from_world;

            init_gpu_camera<<<1, 1>>>(renderer, resolution, camera_transform);
            return;
        }

        std::cerr << "Camera type `" << camera_type << "` not implemented\n";
        throw std::runtime_error("camera type not implemented");
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
        auto render_from_object = get_render_from_object();
        auto object_from_render = render_from_object.inverse();
        bool reverse_orientation = graphics_state.reversed_orientation;

        // TODO: progress 2024/01/25 wentao implementing Shape
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
            resolution = Point2i(1368, 1026);
            printf("resolution 1368x1026 is hardcoded, please delete me\n");
            // TODO: read resolution from PBRT file

            checkCudaErrors(cudaMallocManaged((void **)&renderer, sizeof(Renderer)));
            init_gpu_renderer<<<1, 1>>>(renderer);

            option_lookat();
            option_camera();
            init_gpu_integrator<<<1, 1>>>(renderer);

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

        // TODO: progress 2024/01/23 parsing PBRT file
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

        free_renderer<<<1, 1>>>(renderer);
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

        // TODO: read those parameter from PBRT file
        builder.render(1, "output.png");
    }

    void render(int num_samples, const std::string &file_name) const {
        // TODO: read those parameter from PBRT file
        int thread_width = 8;
        int thread_height = 8;

        std::cerr << "Rendering a " << resolution.x << "x" << resolution.y
                  << " image (samples per pixel: " << num_samples << ") ";
        std::cerr << "in " << thread_width << "x" << thread_height << " blocks.\n";

        // TODO: compute num_primitive from PBRT file
        init_gpu_aggregate<<<1, 1>>>(renderer, 6);

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // allocate FB
        Color *frame_buffer;
        checkCudaErrors(
            cudaMallocManaged((void **)&frame_buffer, sizeof(Color) * resolution.x * resolution.y));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        clock_t start = clock();
        dim3 blocks(resolution.x / thread_width + 1, resolution.y / thread_height + 1, 1);
        dim3 threads(thread_width, thread_height, 1);

        gpu_render<<<blocks, threads>>>(frame_buffer, num_samples, renderer);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        const double timer_seconds = (double)(clock() - start) / CLOCKS_PER_SEC;
        std::cerr << std::fixed << std::setprecision(1) << "took " << timer_seconds
                  << " seconds.\n";

        writer_to_file(file_name, frame_buffer, resolution.x, resolution.y);

        checkCudaErrors(cudaFree(frame_buffer));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        std::cout << "image saved to `" << file_name << "`\n";
    }
};
