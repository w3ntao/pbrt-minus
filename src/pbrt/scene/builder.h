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
#include "pbrt/shapes/loop_subdivide.h"

#include "pbrt/gpu/rendering.cuh"

namespace {
std::string get_dirname(const std::string &full_path) {
    const size_t last_slash_idx = full_path.rfind('/');
    if (std::string::npos != last_slash_idx) {
        return full_path.substr(0, last_slash_idx);
    }

    throw std::runtime_error("get_dirname() fails");
}

int compute_minimum_blocks(int total_jobs, int batch) {
    return int(std::ceil(float(total_jobs) / float(batch)));
}

int integer_divide_round_up(int dividen, int divisor) {
    return int(std::ceil(float(dividen) / float(divisor)));
}

} // namespace

class GraphicsState {
  public:
    GraphicsState() : current_transform(Transform::identity()), reverse_orientation(false) {}

    Transform current_transform;
    bool reverse_orientation;
};

struct GPUconstants {
    GPUconstants() {
        auto start = std::chrono::system_clock::now();

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

        int scale_size = sizeof(double) * RGBtoSpectrumData::RES;
        int coeffs_size = sizeof(double) * 3 * 3 * RGBtoSpectrumData::RES * RGBtoSpectrumData::RES *
                          RGBtoSpectrumData::RES;

        checkCudaErrors(cudaMallocManaged((void **)&rgb_to_spectrum_table_scale, scale_size));
        checkCudaErrors(
            cudaMallocManaged((void **)&rgb_to_spectrum_table_coefficients, coeffs_size));

        checkCudaErrors(cudaMemcpy(rgb_to_spectrum_table_scale, rgb_spectrum_table_cpu.z_nodes,
                                   scale_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(rgb_to_spectrum_table_coefficients,
                                   rgb_spectrum_table_cpu.coefficients, coeffs_size,
                                   cudaMemcpyHostToDevice));

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

        const std::chrono::duration<double> duration{std::chrono::system_clock::now() - start};
        std::cout << std::fixed << std::setprecision(1) << "spectra computing took "
                  << duration.count() << " seconds.\n"
                  << std::flush;
    }

    ~GPUconstants() {
        for (auto ptr : std::vector<void *>{
                 cie_lambdas_gpu,
                 cie_x_value_gpu,
                 cie_y_value_gpu,
                 cie_z_value_gpu,
                 cie_illum_d6500_gpu,
                 cie_s_lambda_gpu,
                 cie_s0_gpu,
                 cie_s1_gpu,
                 cie_s2_gpu,
                 rgb_to_spectrum_table_gpu,
             }) {
            checkCudaErrors(cudaFree(ptr));
        }

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
    std::string root;
    std::optional<int> samples_per_pixel;
    std::optional<std::string> integrator_name;

    GPU::Renderer *renderer = nullptr;
    GPUconstants gpu_constants;
    std::vector<Shape *> gpu_primitives;
    std::vector<void *> gpu_dynamic_pointers;

    std::optional<Point2i> film_resolution = std::nullopt;
    std::string filename;
    std::vector<Token> camera_tokens;
    std::vector<Token> film_tokens;
    std::vector<Token> sampler_tokens;

    GraphicsState graphics_state;
    std::stack<GraphicsState> pushed_graphics_state;
    std::map<std::string, Transform> named_coordinate_systems;
    Transform render_from_world;

    explicit SceneBuilder(const CommandLineOption &command_line_option)
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
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(renderer));

        for (auto ptr : gpu_dynamic_pointers) {
            checkCudaErrors(cudaFree(ptr));
        }
        checkCudaErrors(cudaGetLastError());
    }

    static std::vector<int> group_tokens(const std::vector<Token> &tokens) {
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

        gpu_init_pixel_sensor_cie_1931<<<1, 1>>>(renderer, gpu_constants.cie_s0_gpu,
                                                 gpu_constants.cie_s1_gpu, gpu_constants.cie_s2_gpu,
                                                 gpu_constants.cie_s_lambda_gpu);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        gpu_init_filter<<<1, 1>>>(renderer);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        Pixel *gpu_pixels;
        checkCudaErrors(cudaMallocManaged((void **)&gpu_pixels,
                                          sizeof(Pixel) * film_resolution->x * film_resolution->y));
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

    void parse_lookat(const std::vector<Token> &tokens) {
        if (tokens[0] != Token(Keyword, "LookAt")) {
            throw std::runtime_error("expect Keyword(LookAt)");
        }

        std::vector<double> data;
        for (int idx = 1; idx < tokens.size(); idx++) {
            data.push_back(tokens[idx].to_number());
        }

        auto position = Point3f(data[0], data[1], data[2]);
        auto look = Point3f(data[3], data[4], data[5]);
        auto up = Vector3f(data[6], data[7], data[8]);

        graphics_state.current_transform =
            graphics_state.current_transform * Transform::lookat(position, look, up);
    }

    void parse_rotate(const std::vector<Token> &tokens) {
        if (tokens[0] != Token(Keyword, "Rotate")) {
            throw std::runtime_error("expect Keyword(Rotate)");
        }

        std::vector<double> data;
        for (int idx = 1; idx < tokens.size(); idx++) {
            data.push_back(tokens[idx].to_number());
        }

        graphics_state.current_transform = graphics_state.current_transform *
                                           Transform::rotate(data[0], data[1], data[2], data[3]);
    }

    void parse_scale(const std::vector<Token> &tokens) {
        if (tokens[0] != Token(Keyword, "Scale")) {
            throw std::runtime_error("expect Keyword(Scale)");
        }

        std::vector<double> data;
        for (int idx = 1; idx < tokens.size(); idx++) {
            data.push_back(tokens[idx].to_number());
        }

        graphics_state.current_transform *= Transform::scale(data[0], data[1], data[2]);
    }

    void world_shape(const std::vector<Token> &tokens) {
        if (tokens[0] != Token(Keyword, "Shape")) {
            throw std::runtime_error("expect Keyword(Shape)");
        }

        auto render_from_object = get_render_from_object();
        bool reverse_orientation = graphics_state.reverse_orientation;

        const auto parameters = ParameterDict(tokens);

        auto type_of_shape = tokens[1].value[0];
        if (type_of_shape == "trianglemesh") {
            auto uv = parameters.get_point2("uv");
            auto indices = parameters.get_integer("indices");
            auto points = parameters.get_point3("P");

            add_triangle_mesh(points, indices, uv, reverse_orientation, render_from_object);

            return;
        }

        if (type_of_shape == "loopsubdiv") {
            auto levels = parameters.get_integer("levels")[0];
            auto indices = parameters.get_integer("indices");
            auto points = parameters.get_point3("P");

            const auto loop_subdivide_data = LoopSubdivide::build(levels, indices, points);

            add_triangle_mesh(loop_subdivide_data.p_limit, loop_subdivide_data.vertex_indices,
                              std::vector<Point2f>(), reverse_orientation, render_from_object);

            return;
        }

        if (type_of_shape == "sphere") {
            printf("\nignore Shape::sphere for the moment\n\n");
            return;
        }

        std::cerr << "\n\nunknown shape: `" << type_of_shape << "`\n\n\n";
        throw std::runtime_error("unknown shape");
    }

    void world_translate(const std::vector<Token> &tokens) {
        std::vector<double> data;
        for (int idx = 1; idx < tokens.size(); idx++) {
            data.push_back(tokens[idx].to_number());
        }

        graphics_state.current_transform *= Transform::translate(data[0], data[1], data[2]);
    }

    void add_triangle_mesh(const std::vector<Point3f> &points, const std::vector<int> &indices,
                           const std::vector<Point2f> &uv, bool reverse_orientation,
                           const Transform &render_from_object) {
        Point3f *gpu_points;
        checkCudaErrors(cudaMallocManaged((void **)&gpu_points, sizeof(Point3f) * points.size()));
        checkCudaErrors(cudaMemcpy(gpu_points, points.data(), sizeof(Point3f) * points.size(),
                                   cudaMemcpyHostToDevice));

        int batch = 256;
        int total_jobs = points.size() / batch + 1;
        GPU::apply_transform<<<total_jobs, batch>>>(gpu_points, render_from_object, points.size());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        int *gpu_indices;
        checkCudaErrors(cudaMallocManaged((void **)&gpu_indices, sizeof(int) * indices.size()));
        checkCudaErrors(cudaMemcpy(gpu_indices, indices.data(), sizeof(int) * indices.size(),
                                   cudaMemcpyHostToDevice));

        Point2f *gpu_uv = nullptr;
        if (!uv.empty()) {
            checkCudaErrors(cudaMallocManaged((void **)&gpu_uv, sizeof(Point2f) * uv.size()));
            checkCudaErrors(
                cudaMemcpy(gpu_uv, uv.data(), sizeof(Point2f) * uv.size(), cudaMemcpyHostToDevice));
        }

        TriangleMesh *gpu_mesh;
        checkCudaErrors(cudaMallocManaged((void **)&gpu_mesh, sizeof(TriangleMesh)));

        *gpu_mesh = TriangleMesh(reverse_orientation, gpu_indices, indices.size(), gpu_points,
                                 points.size());

        gpu_add_triangle_mesh<<<1, 1>>>(renderer, gpu_mesh);

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        gpu_dynamic_pointers.push_back(gpu_indices);
        gpu_dynamic_pointers.push_back(gpu_points);
        gpu_dynamic_pointers.push_back(gpu_uv);
        gpu_dynamic_pointers.push_back(gpu_mesh);
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
            option_film();
            option_camera();
            option_sampler();

            gpu_init_aggregate<<<1, 1>>>(renderer);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

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

            if (keyword == "Include") {
                auto subfile = tokens[1].value[0];
                auto fullpath = !root.empty() ? root + "/" + subfile : subfile;
                parse_file(fullpath);
                return;
            }

            if (keyword == "LookAt") {
                parse_lookat(tokens);
                return;
            }

            if (keyword == "Rotate") {
                parse_rotate(tokens);
                return;
            }

            if (keyword == "Sampler") {
                sampler_tokens = tokens;
                return;
            }

            if (keyword == "Scale") {
                parse_scale(tokens);
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

            if (keyword == "AreaLightSource" || keyword == "Integrator" ||
                keyword == "LightSource" || keyword == "Material" ||
                keyword == "MakeNamedMaterial" || keyword == "NamedMaterial" ||
                keyword == "ReverseOrientation") {
                std::cout << "parse_tokens::Keyword `" << keyword << "` ignored for the moment\n";
                return;
            }

            std::cerr << "\nERROR: parse_tokens::Keyword `" << keyword << "` not implemented\n\n";
            throw std::runtime_error("parse_tokens(): unknown keyword");
        }

        default: {
            printf("unknown token type: `%d`\n", first_token.type);
            throw std::runtime_error("parse_tokens() fail");
        }
        }
    }

    void parse_file(const std::string &filename) {
        const auto all_tokens = parse_pbrt_into_token(filename);
        const auto range_of_tokens = SceneBuilder::group_tokens(all_tokens);

        for (int range_idx = 0; range_idx < range_of_tokens.size() - 1; ++range_idx) {
            auto current_tokens = std::vector(all_tokens.begin() + range_of_tokens[range_idx],
                                              all_tokens.begin() + range_of_tokens[range_idx + 1]);

            parse_tokens(current_tokens);
        }
    }

    void preprocess() {
        auto start_bvh = std::chrono::system_clock::now();

        int *gpu_shape_num;
        checkCudaErrors(cudaMallocManaged((void **)&gpu_shape_num, sizeof(int)));
        gpu_get_primitive_num<<<1, 1>>>(renderer, gpu_shape_num);
        printf("shape num: %d\n", *gpu_shape_num);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        gpu_dynamic_pointers.push_back(gpu_shape_num);

        checkCudaErrors(cudaMallocManaged((void **)&(renderer->hlbvh), sizeof(HLBVH)));

        BVHPrimitive *gpu_bvh_primitives;
        checkCudaErrors(cudaMallocManaged((void **)&gpu_bvh_primitives,
                                          sizeof(BVHPrimitive) * (*gpu_shape_num)));

        MortonPrimitive *gpu_morton_primitives;
        checkCudaErrors(cudaMallocManaged((void **)&gpu_morton_primitives,
                                          sizeof(MortonPrimitive) * (*gpu_shape_num)));

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        gpu_dynamic_pointers.push_back(renderer->hlbvh);
        gpu_dynamic_pointers.push_back(gpu_bvh_primitives);
        gpu_dynamic_pointers.push_back(gpu_morton_primitives);

        gpu_create_hlbvh<<<1, 1>>>(renderer, renderer->hlbvh, gpu_bvh_primitives,
                                   gpu_morton_primitives);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        {
            int threads = 128;
            int blocks =
                integer_divide_round_up(std::max(*gpu_shape_num, MAX_TREELET_NUM), threads);
            gpu_hlbvh_init_bvh_primitives_and_treelets<<<blocks, threads>>>(renderer);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }

        gpu_hlbvh_compute_full_bounds<<<1, 1>>>(renderer);
        // TODO: paralellize this later
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        {
            int batch_size = 128;
            int total_job = std::max(*gpu_shape_num, MAX_TREELET_NUM);
            int blocks = integer_divide_round_up(total_job, batch_size);
            gpu_hlbvh_compute_morton_code<<<blocks, batch_size>>>(renderer);

            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }

        {
            struct {
                bool operator()(const MortonPrimitive &left, const MortonPrimitive &right) const {
                    return (left.morton_code & TREELET_MASK) < (right.morton_code & TREELET_MASK);
                }
            } morton_comparator;

            std::sort(renderer->hlbvh->morton_primitives,
                      renderer->hlbvh->morton_primitives + *gpu_shape_num, morton_comparator);
            // TODO: rewrite this sorting in GPU
        }

        // init top treelets
        {
            int threads = 64;
            int blocks = integer_divide_round_up(*gpu_shape_num, threads);
            gpu_hlbvh_build_treelets<<<blocks, threads>>>(renderer);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }

        auto hlbvh = renderer->hlbvh;
        std::vector<int> treelet_indices;
        int max_primitive_num_in_a_treelet = 0;
        int primitives_counter = 0;
        for (int idx = 0; idx < MAX_TREELET_NUM; idx++) {
            int n_primitives = hlbvh->treelets[idx].n_primitives;
            if (n_primitives <= 0) {
                continue;
            }

            primitives_counter += n_primitives;

            max_primitive_num_in_a_treelet = std::max(max_primitive_num_in_a_treelet, n_primitives);
            treelet_indices.push_back(idx);
        }

        assert(primitives_counter == *gpu_shape_num);

        printf("HLBVH: %zu/%d treelets filled (max primitives in a treelet: %d)\n",
               treelet_indices.size(), MAX_TREELET_NUM, max_primitive_num_in_a_treelet);

        BVHBuildNodeForTreelet *build_nodes_for_treelets;
        checkCudaErrors(
            cudaMallocManaged((void **)&build_nodes_for_treelets,
                              sizeof(BVHBuildNodeForTreelet) * (2 * treelet_indices.size() + 1)));

        hlbvh->build_nodes_for_treelets = build_nodes_for_treelets;
        gpu_dynamic_pointers.push_back(build_nodes_for_treelets);

        hlbvh->build_bvh_for_treelets();

        const std::chrono::duration<double> duration_bvh{std::chrono::system_clock::now() -
                                                         start_bvh};
        std::cout << std::fixed << std::setprecision(1) << "BVH constructing took "
                  << duration_bvh.count() << " seconds.\n"
                  << std::flush;
    }

    void render() const {
        auto start_rendering = std::chrono::system_clock::now();

        int thread_width = 8;
        int thread_height = 8;

        std::cout << "\n";
        std::cout << "rendering a " << film_resolution->x << "x" << film_resolution->y
                  << " image (samples per pixel: " << samples_per_pixel.value() << ") ";
        std::cout << "in " << thread_width << "x" << thread_height << " blocks.\n" << std::flush;

        dim3 blocks(film_resolution->x / thread_width + 1, film_resolution->y / thread_height + 1,
                    1);
        dim3 threads(thread_width, thread_height, 1);
        gpu_parallel_render<<<blocks, threads>>>(renderer, samples_per_pixel.value());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        const std::chrono::duration<double> duration_rendering{std::chrono::system_clock::now() -
                                                               start_rendering};

        std::cout << std::fixed << std::setprecision(1) << "rendering took "
                  << duration_rendering.count() << " seconds.\n"
                  << std::flush;

        RGB *output_rgb;
        checkCudaErrors(cudaMallocManaged((void **)&output_rgb,
                                          sizeof(RGB) * film_resolution->x * film_resolution->y));

        copy_gpu_pixels_to_rgb<<<blocks, threads>>>(renderer, output_rgb);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

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

        auto builder = SceneBuilder(command_line_option);

        auto input_file = command_line_option.input_file;

        builder.root = get_dirname(input_file);

        builder.parse_file(input_file);

        builder.preprocess();

        builder.render();
    }
};
