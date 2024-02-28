#pragma once

#include <stack>
#include <map>
#include <chrono>

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/transform.h"

#include "pbrt/spectra/rgb_to_spectrum_data.h"
#include "pbrt/scene/command_line_option.h"
#include "pbrt/scene/parser.h"
#include "pbrt/scene/parameter_dict.h"

#include "pbrt/gpu/rendering.cuh"

class GraphicsState {
  public:
    GraphicsState() : current_transform(Transform::identity()), reverse_orientation(false) {}

    Transform current_transform;
    bool reverse_orientation;
};

struct GPUconstants {
    GPUconstants() {
        checkCudaErrors(cudaMallocManaged((void **)&cie_lambdas_gpu, sizeof(CIE_LAMBDA_CPU)));
        checkCudaErrors(cudaMallocManaged((void **)&cie_x_value_gpu, sizeof(CIE_X_VALUE_CPU)));
        checkCudaErrors(cudaMallocManaged((void **)&cie_y_value_gpu, sizeof(CIE_Y_VALUE_CPU)));
        checkCudaErrors(cudaMallocManaged((void **)&cie_z_value_gpu, sizeof(CIE_Z_VALUE_CPU)));
        checkCudaErrors(cudaMallocManaged((void **)&cie_illum_d6500_gpu, sizeof(CIE_Illum_D6500)));

        checkCudaErrors(cudaMallocManaged((void **)&cie_s0_gpu, sizeof(CIE_S0)));
        checkCudaErrors(cudaMallocManaged((void **)&cie_s1_gpu, sizeof(CIE_S1)));
        checkCudaErrors(cudaMallocManaged((void **)&cie_s2_gpu, sizeof(CIE_S2)));
        checkCudaErrors(cudaMallocManaged((void **)&cie_s_lambda_gpu, sizeof(CIE_S_lambda)));

        checkCudaErrors(cudaMemcpy(cie_lambdas_gpu, CIE_LAMBDA_CPU, sizeof(CIE_LAMBDA_CPU),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(cie_x_value_gpu, CIE_X_VALUE_CPU, sizeof(CIE_X_VALUE_CPU),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(cie_y_value_gpu, CIE_Y_VALUE_CPU, sizeof(CIE_Y_VALUE_CPU),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(cie_z_value_gpu, CIE_Z_VALUE_CPU, sizeof(CIE_Z_VALUE_CPU),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(cie_illum_d6500_gpu, CIE_Illum_D6500, sizeof(CIE_Illum_D6500),
                                   cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(cie_s0_gpu, CIE_S0, sizeof(CIE_S0), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(cie_s1_gpu, CIE_S1, sizeof(CIE_S1), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(cie_s2_gpu, CIE_S2, sizeof(CIE_S2), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(cie_s_lambda_gpu, CIE_S_lambda, sizeof(CIE_S_lambda),
                                   cudaMemcpyHostToDevice));

        const auto rgb_spectrum_table_cpu = RGBtoSpectrumData::compute_spectrum_table_data("sRGB");

        double *rgb_to_spectrum_table_scale;
        double *rgb_to_spectrum_table_coefficients;

        checkCudaErrors(cudaMallocManaged((void **)&rgb_to_spectrum_table_scale,
                                          sizeof(double) * rgb_spectrum_table_cpu.z_nodes.size()));
        checkCudaErrors(
            cudaMallocManaged((void **)&rgb_to_spectrum_table_coefficients,
                              sizeof(double) * rgb_spectrum_table_cpu.coefficients.size()));

        checkCudaErrors(cudaMemcpy(
            rgb_to_spectrum_table_scale, rgb_spectrum_table_cpu.z_nodes.data(),
            sizeof(double) * rgb_spectrum_table_cpu.z_nodes.size(), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(
            rgb_to_spectrum_table_coefficients, rgb_spectrum_table_cpu.coefficients.data(),
            sizeof(double) * rgb_spectrum_table_cpu.coefficients.size(), cudaMemcpyHostToDevice));

        // TODO: add rgb_to_spectrum_table_gpu to GPUconstants
        checkCudaErrors(cudaMallocManaged((void **)&rgb_to_spectrum_table_gpu,
                                          sizeof(RGBtoSpectrumData::RGBtoSpectrumTableGPU)));

        const int num_component = 3;
        const int rgb_to_spectrum_data_resolution = RGBtoSpectrumData::RES;
        const int channel = 3;

        /*
         * max thread size: 1024
         * total dimension: 3 * 64 * 64 * 64 * 3
         * 3: blocks.x
         * 64: blocks.y
         * 64: blocks.z
         * 64: threads.x
         * 3:  threads.y
         */
        dim3 blocks(num_component, rgb_to_spectrum_data_resolution,
                    rgb_to_spectrum_data_resolution);
        dim3 threads(rgb_to_spectrum_data_resolution, channel, 1);
        GPU::gpu_init_rgb_to_spectrum_table_coefficients<<<blocks, threads>>>(
            rgb_to_spectrum_table_gpu, rgb_to_spectrum_table_coefficients);

        GPU::gpu_init_rgb_to_spectrum_table_scale<<<1, rgb_to_spectrum_data_resolution>>>(
            rgb_to_spectrum_table_gpu, rgb_to_spectrum_table_scale);

        for (auto ptr : {rgb_to_spectrum_table_scale, rgb_to_spectrum_table_coefficients}) {
            checkCudaErrors(cudaFree(ptr));
        }
    }

    ~GPUconstants() {
        for (auto ptr : {
                 cie_lambdas_gpu,
                 cie_x_value_gpu,
                 cie_y_value_gpu,
                 cie_z_value_gpu,
                 cie_illum_d6500_gpu,
                 cie_s_lambda_gpu,
                 cie_s0_gpu,
                 cie_s1_gpu,
                 cie_s2_gpu,
             }) {
            checkCudaErrors(cudaFree(ptr));
        }
        checkCudaErrors(cudaFree(rgb_to_spectrum_table_gpu));

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    double *cie_lambdas_gpu = nullptr;
    double *cie_x_value_gpu = nullptr;
    double *cie_y_value_gpu = nullptr;
    double *cie_z_value_gpu = nullptr;

    double *cie_illum_d6500_gpu = nullptr;

    double *cie_s_lambda_gpu = nullptr;
    double *cie_s0_gpu = nullptr;
    double *cie_s1_gpu = nullptr;
    double *cie_s2_gpu = nullptr;

    RGBtoSpectrumData::RGBtoSpectrumTableGPU *rgb_to_spectrum_table_gpu = nullptr;
};

class SceneBuilder {
    std::optional<int> samples_per_pixel;
    std::optional<std::string> integrator_name;

    GPU::Renderer *renderer = nullptr;
    GPUconstants gpu_constants;
    std::vector<void *> gpu_dynamic_pointers;

    std::optional<Point2i> film_resolution = std::nullopt;
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
          integrator_name(command_line_option.integrator) {

        checkCudaErrors(cudaMallocManaged((void **)&renderer, sizeof(GPU::Renderer)));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        gpu_init_global_variables<<<1, 1>>>(
            renderer, gpu_constants.cie_lambdas_gpu, gpu_constants.cie_x_value_gpu,
            gpu_constants.cie_y_value_gpu, gpu_constants.cie_z_value_gpu,
            gpu_constants.cie_illum_d6500_gpu, sizeof(CIE_Illum_D6500) / sizeof(double),
            gpu_constants.rgb_to_spectrum_table_gpu);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    ~SceneBuilder() {
        gpu_free_renderer<<<1, 1>>>(renderer);
        checkCudaErrors(cudaFree(renderer));

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        for (auto ptr : gpu_dynamic_pointers) {
            checkCudaErrors(cudaFree(ptr));
        }
    }

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

            gpu_init_camera<<<1, 1>>>(renderer, film_resolution.value(), camera_transform, fov);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            return;
        }

        std::cerr << "Camera type `" << camera_type << "` not implemented\n";
        throw std::runtime_error("camera type not implemented");
    }

    void option_film() {
        auto parameters = ParameterDict(film_tokens);

        auto _resolution_x = parameters.get_integer("xresolution")[0];
        auto _resolution_y = parameters.get_integer("yresolution")[0];

        film_resolution = Point2i(_resolution_x, _resolution_y);
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

            if (!render_from_object.is_identity()) {
                for (auto &_p : points) {
                    _p = render_from_object(_p);
                }
            }

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

            gpu_dynamic_pointers.push_back(gpu_indicies);
            gpu_dynamic_pointers.push_back(gpu_points);
            gpu_dynamic_pointers.push_back(gpu_uv);

            gpu_add_triangle_mesh<<<1, 1>>>(renderer, reverse_orientation, gpu_points,
                                            points.size(), gpu_indicies, indices.size(), gpu_uv,
                                            uv.size());

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
            option_lookat();
            option_film();
            option_camera();
            option_sampler();

            gpu_init_pixel_sensor_cie_1931<<<1, 1>>>(
                renderer, gpu_constants.cie_s0_gpu, gpu_constants.cie_s1_gpu,
                gpu_constants.cie_s2_gpu, gpu_constants.cie_s_lambda_gpu);
            gpu_init_filter<<<1, 1>>>(renderer);

            Pixel *gpu_pixels;
            checkCudaErrors(cudaMallocManaged(
                (void **)&gpu_pixels, sizeof(Pixel) * film_resolution->x * film_resolution->y));
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            gpu_dynamic_pointers.push_back(gpu_pixels);

            int batch = 256;
            int total_job = film_resolution->x * film_resolution->y / 256 + 1;
            GPU::gpu_init_pixels<<<total_job, batch>>>(gpu_pixels, film_resolution.value());
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            gpu_init_rgb_film<<<1, 1>>>(renderer, film_resolution.value(), gpu_pixels);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            gpu_init_aggregate<<<1, 1>>>(renderer);
            gpu_init_integrator<<<1, 1>>>(renderer);

            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

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

    void render() const {
        int thread_width = 8;
        int thread_height = 8;

        std::cout << "\n";
        std::cout << "rendering a " << film_resolution->x << "x" << film_resolution->y
                  << " image (samples per pixel: " << samples_per_pixel.value() << ") ";
        std::cout << "in " << thread_width << "x" << thread_height << " blocks.\n";

        gpu_aggregate_preprocess<<<1, 1>>>(renderer);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        auto start = std::chrono::system_clock::now();

        dim3 blocks(film_resolution->x / thread_width + 1, film_resolution->y / thread_height + 1,
                    1);
        dim3 threads(thread_width, thread_height, 1);
        gpu_parallel_render<<<blocks, threads>>>(renderer, samples_per_pixel.value());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        const std::chrono::duration<double> duration{std::chrono::system_clock::now() - start};

        std::cout << std::fixed << std::setprecision(1) << "took " << duration.count()
                  << " seconds.\n";

        RGB *output_rgb;
        checkCudaErrors(cudaMallocManaged((void **)&output_rgb,
                                          sizeof(RGB) * film_resolution->x * film_resolution->y));

        int batch = 256;
        int total_jobs = film_resolution->x * film_resolution->y / batch + 1;
        write_frame_buffer_to_rgb<<<total_jobs, batch>>>(renderer, output_rgb);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // TODO: directly access RGB contents from Renderer?
        GPU::writer_to_file(filename, output_rgb, film_resolution.value());

        checkCudaErrors(cudaFree(output_rgb));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        std::cout << "image saved to `" << filename << "`\n";
    }

  public:
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
};
