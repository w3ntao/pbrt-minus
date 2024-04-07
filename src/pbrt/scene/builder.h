#pragma once

#include <stack>
#include <map>
#include <chrono>

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/transform.h"

#include "pbrt/filters/box.h"

#include "pbrt/integrators/ambient_occlusion.h"
#include "pbrt/integrators/surface_normal.h"

#include "pbrt/spectra/rgb_to_spectrum_data.h"
#include "pbrt/scene/command_line_option.h"
#include "pbrt/scene/parser.h"
#include "pbrt/scene/parameter_dict.h"
#include "pbrt/shapes/loop_subdivide.h"

#include "pbrt/gpu/renderer.h"

namespace {
std::string get_dirname(const std::string &full_path) {
    const size_t last_slash_idx = full_path.rfind('/');
    if (std::string::npos != last_slash_idx) {
        return full_path.substr(0, last_slash_idx);
    }

    throw std::runtime_error("get_dirname() fails");
}
} // namespace

class GraphicsState {
  public:
    GraphicsState() : current_transform(Transform::identity()), reverse_orientation(false) {}

    Transform current_transform;
    bool reverse_orientation;
};

struct PreComputedSpectrum {
    explicit PreComputedSpectrum(ThreadPool &thread_pool) {
        auto start = std::chrono::system_clock::now();

        for (uint idx = 0; idx < 3; idx++) {
            checkCudaErrors(
                cudaMallocManaged((void **)&(dense_cie_xyz[idx]), sizeof(DenselySampledSpectrum)));
        }
        dense_cie_xyz[0]->init_from_pls_lambdas_values(CIE_LAMBDA_CPU, CIE_X_VALUE_CPU,
                                                       NUM_CIE_SAMPLES);
        dense_cie_xyz[1]->init_from_pls_lambdas_values(CIE_LAMBDA_CPU, CIE_Y_VALUE_CPU,
                                                       NUM_CIE_SAMPLES);
        dense_cie_xyz[2]->init_from_pls_lambdas_values(CIE_LAMBDA_CPU, CIE_Z_VALUE_CPU,
                                                       NUM_CIE_SAMPLES);

        for (uint idx = 0; idx < 3; idx++) {
            Spectrum *temp_spectrum;
            checkCudaErrors(cudaMallocManaged((void **)&temp_spectrum, sizeof(Spectrum)));
            temp_spectrum->init(dense_cie_xyz[idx]);
            cie_xyz[idx] = temp_spectrum;
        }

        checkCudaErrors(
            cudaMallocManaged((void **)&dense_illum_d65, sizeof(DenselySampledSpectrum)));
        checkCudaErrors(cudaMallocManaged((void **)&illum_d65, sizeof(Spectrum)));

        dense_illum_d65->init_from_pls_interleaved_samples(
            CIE_Illum_D6500, sizeof(CIE_Illum_D6500) / sizeof(CIE_Illum_D6500[0]), true,
            cie_xyz[1]);
        illum_d65->init(dense_illum_d65);

        checkCudaErrors(cudaMallocManaged((void **)&rgb_to_spectrum_table,
                                          sizeof(RGBtoSpectrumData::RGBtoSpectrumTable)));

        rgb_to_spectrum_table->init("sRGB", thread_pool);

        const std::chrono::duration<double> duration{std::chrono::system_clock::now() - start};
        std::cout << std::fixed << std::setprecision(1) << "spectra computing took "
                  << duration.count() << " seconds.\n"
                  << std::flush;
    }

    ~PreComputedSpectrum() {
        for (auto ptr : std::vector<void *>{
                 rgb_to_spectrum_table,
                 dense_illum_d65,
                 illum_d65,
             }) {
            checkCudaErrors(cudaFree(ptr));
        }

        for (uint idx = 0; idx < 3; idx++) {
            checkCudaErrors(cudaFree(dense_cie_xyz[idx]));
            checkCudaErrors(cudaFree((void *)cie_xyz[idx]));
        }

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    const Spectrum *cie_xyz[3] = {nullptr};
    Spectrum *illum_d65 = nullptr;

    RGBtoSpectrumData::RGBtoSpectrumTable *rgb_to_spectrum_table = nullptr;

  private:
    DenselySampledSpectrum *dense_cie_xyz[3] = {nullptr};
    DenselySampledSpectrum *dense_illum_d65 = nullptr;
};

class SceneBuilder {
  private:
    ThreadPool thread_pool;
    std::string root;

  public:
    std::optional<int> samples_per_pixel;
    std::optional<std::string> integrator_name;

    GPU::Renderer *renderer = nullptr;

    PreComputedSpectrum pre_computed_spectrum;
    std::vector<void *> gpu_dynamic_pointers;
    std::vector<const Shape *> gpu_primitives;

    std::optional<Point2i> film_resolution = std::nullopt;
    std::string output_filename;
    std::vector<Token> camera_tokens;
    std::vector<Token> film_tokens;
    std::vector<Token> sampler_tokens;

    GraphicsState graphics_state;
    std::stack<GraphicsState> pushed_graphics_state;
    std::map<std::string, Transform> named_coordinate_systems;
    Transform render_from_world;

    explicit SceneBuilder(const CommandLineOption &command_line_option)
        : samples_per_pixel(command_line_option.samples_per_pixel),
          pre_computed_spectrum(PreComputedSpectrum(thread_pool)) {

        GPU::GlobalVariable *global_variables;
        checkCudaErrors(cudaMallocManaged((void **)&global_variables, sizeof(GPU::GlobalVariable)));

        checkCudaErrors(cudaMallocManaged((void **)&(global_variables->rgb_color_space),
                                          sizeof(RGBColorSpace)));

        global_variables->init(pre_computed_spectrum.cie_xyz, pre_computed_spectrum.illum_d65,
                               pre_computed_spectrum.rgb_to_spectrum_table,
                               RGBtoSpectrumData::Gamut::sRGB);

        checkCudaErrors(cudaMallocManaged((void **)&renderer, sizeof(GPU::Renderer)));
        renderer->global_variables = global_variables;

        checkCudaErrors(cudaMallocManaged((void **)&(renderer->bvh), sizeof(HLBVH)));
        checkCudaErrors(cudaMallocManaged((void **)&(renderer->camera), sizeof(Camera)));
        checkCudaErrors(cudaMallocManaged((void **)&(renderer->film), sizeof(Film)));
        checkCudaErrors(cudaMallocManaged((void **)&(renderer->filter), sizeof(Filter)));
        checkCudaErrors(cudaMallocManaged((void **)&(renderer->integrator), sizeof(Integrator)));

        gpu_dynamic_pointers.push_back(global_variables);
        gpu_dynamic_pointers.push_back(global_variables->rgb_color_space);

        gpu_dynamic_pointers.push_back(renderer);
        gpu_dynamic_pointers.push_back(renderer->bvh);
        gpu_dynamic_pointers.push_back(renderer->camera);
        gpu_dynamic_pointers.push_back(renderer->film);
        gpu_dynamic_pointers.push_back(renderer->filter);
        gpu_dynamic_pointers.push_back(renderer->integrator);
    }

    ~SceneBuilder() {
        auto start = std::chrono::system_clock::now();

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        for (auto ptr : gpu_dynamic_pointers) {
            checkCudaErrors(cudaFree(ptr));
        }
        checkCudaErrors(cudaGetLastError());

        const std::chrono::duration<double> duration{std::chrono::system_clock::now() - start};

        std::cout << std::fixed << std::setprecision(1) << "GPU resource release took "
                  << duration.count() << " seconds.\n"
                  << std::flush;
    }

    static std::vector<uint> group_tokens(const std::vector<Token> &tokens) {
        std::vector<uint> keyword_range;
        for (int idx = 0; idx < tokens.size(); ++idx) {
            const auto &token = tokens[idx];
            if (token.type == TokenType::WorldBegin || token.type == TokenType::AttributeBegin ||
                token.type == TokenType::AttributeEnd || token.type == TokenType::Keyword) {
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
        const auto camera_type = camera_tokens[1].values[0];
        if (camera_type == "perspective") {
            auto camera_from_world = graphics_state.current_transform;
            auto world_from_camera = camera_from_world.inverse();

            named_coordinate_systems["camera"] = world_from_camera;

            auto camera_transform = CameraTransform(
                world_from_camera, RenderingCoordinateSystem::CameraWorldCoordSystem);

            render_from_world = camera_transform.render_from_world;

            double fov = 90;
            if (const auto _fov = parameters.get_float("fov"); !_fov.empty()) {
                fov = _fov[0];
            }

            PerspectiveCamera *perspective_camera;
            checkCudaErrors(
                cudaMallocManaged((void **)&perspective_camera, sizeof(PerspectiveCamera)));
            gpu_dynamic_pointers.push_back(perspective_camera);

            perspective_camera->init(film_resolution.value(), camera_transform, fov, 0.0);

            renderer->camera->init(perspective_camera);

            return;
        }

        std::cerr << "Camera type `" << camera_type << "` not implemented\n";
        throw std::runtime_error("camera type not implemented");
    }

    void option_filter() {
        BoxFilter *box_filter;
        checkCudaErrors(cudaMallocManaged((void **)&box_filter, sizeof(BoxFilter)));
        gpu_dynamic_pointers.push_back(box_filter);

        box_filter->init(0.5);

        renderer->filter->init(box_filter);
    }

    void option_film() {
        auto parameters = ParameterDict(film_tokens);

        auto _resolution_x = parameters.get_integer("xresolution")[0];
        auto _resolution_y = parameters.get_integer("yresolution")[0];

        film_resolution = Point2i(_resolution_x, _resolution_y);
        output_filename = parameters.get_string("filename");

        DenselySampledSpectrum *_d_illum_dense;
        Spectrum *d_illum;

        checkCudaErrors(
            cudaMallocManaged((void **)&_d_illum_dense, sizeof(DenselySampledSpectrum)));
        checkCudaErrors(cudaMallocManaged((void **)&d_illum, sizeof(Spectrum)));

        gpu_dynamic_pointers.push_back(d_illum);
        gpu_dynamic_pointers.push_back(_d_illum_dense);

        double iso = 100;
        double white_balance_val = 0.0;
        double exposure_time = 1.0;
        double imaging_ratio = exposure_time * iso / 100.0;

        _d_illum_dense->init_cie_d(white_balance_val == 0.0 ? 6500.0 : white_balance_val, CIE_S0,
                                   CIE_S1, CIE_S2, CIE_S_lambda);

        d_illum->init(_d_illum_dense);

        const Spectrum *cie_xyz[3];
        renderer->global_variables->get_cie_xyz(cie_xyz);

        renderer->sensor.init_cie_1931(cie_xyz, renderer->global_variables->rgb_color_space,
                                       white_balance_val == 0 ? nullptr : d_illum, imaging_ratio);

        Pixel *gpu_pixels;
        checkCudaErrors(cudaMallocManaged((void **)&gpu_pixels,
                                          sizeof(Pixel) * film_resolution->x * film_resolution->y));
        gpu_dynamic_pointers.push_back(gpu_pixels);

        {
            uint threads = 1024;
            uint blocks = divide_and_ceil(uint(film_resolution->x * film_resolution->y), threads);
            GPU::init_pixels<<<blocks, threads>>>(gpu_pixels, film_resolution.value());
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }

        RGBFilm *rgb_film;
        checkCudaErrors(cudaMallocManaged((void **)&rgb_film, sizeof(RGBFilm)));
        gpu_dynamic_pointers.push_back(rgb_film);

        rgb_film->init(gpu_pixels, &(renderer->sensor), film_resolution.value(),
                       renderer->global_variables->rgb_color_space);

        renderer->film->init(rgb_film);
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

    void option_integrator() {
        if (!integrator_name.has_value()) {
            integrator_name = "ambientocclusion";
        }

        if (integrator_name == "ambientocclusion") {
            auto illuminant_spectrum = renderer->global_variables->rgb_color_space->illuminant;

            const Spectrum *cie_xyz[3];
            renderer->global_variables->get_cie_xyz(cie_xyz);
            const auto cie_y = cie_xyz[1];
            auto illuminant_scale = 1.0 / illuminant_spectrum->to_photometric(*cie_y);

            AmbientOcclusionIntegrator *ambient_occlusion_integrator;
            checkCudaErrors(cudaMallocManaged((void **)&ambient_occlusion_integrator,
                                              sizeof(AmbientOcclusionIntegrator)));
            gpu_dynamic_pointers.push_back(ambient_occlusion_integrator);

            ambient_occlusion_integrator->init(illuminant_spectrum, illuminant_scale);
            renderer->integrator->init(ambient_occlusion_integrator);

        } else if (integrator_name == "surfacenormal") {
            SurfaceNormalIntegrator *surface_normal_integrator;
            checkCudaErrors(cudaMallocManaged((void **)&surface_normal_integrator,
                                              sizeof(SurfaceNormalIntegrator)));
            gpu_dynamic_pointers.push_back(surface_normal_integrator);

            surface_normal_integrator->init(renderer->global_variables->rgb_color_space);
            renderer->integrator->init(surface_normal_integrator);

        } else {
            const std::string error =
                "parse_tokens(): unknown Integrator name: `" + integrator_name.value() + "`";
            throw std::runtime_error(error.c_str());
        }

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void parse_lookat(const std::vector<Token> &tokens) {
        if (tokens[0] != Token(TokenType::Keyword, "LookAt")) {
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
        if (tokens[0] != Token(TokenType::Keyword, "Rotate")) {
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
        if (tokens[0] != Token(TokenType::Keyword, "Scale")) {
            throw std::runtime_error("expect Keyword(Scale)");
        }

        std::vector<double> data;
        for (int idx = 1; idx < tokens.size(); idx++) {
            data.push_back(tokens[idx].to_number());
        }

        graphics_state.current_transform *= Transform::scale(data[0], data[1], data[2]);
    }

    void world_shape(const std::vector<Token> &tokens) {
        if (tokens[0] != Token(TokenType::Keyword, "Shape")) {
            throw std::runtime_error("expect Keyword(Shape)");
        }

        auto render_from_object = get_render_from_object();
        bool reverse_orientation = graphics_state.reverse_orientation;

        const auto parameters = ParameterDict(tokens);

        auto type_of_shape = tokens[1].values[0];
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

            const auto loop_subdivide_data = LoopSubdivide(levels, indices, points);

            add_triangle_mesh(loop_subdivide_data.p_limit, loop_subdivide_data.vertex_indices,
                              std::vector<Point2f>(), reverse_orientation, render_from_object);

            return;
        }

        if (type_of_shape == "disk" || type_of_shape == "sphere") {
            printf("\nignore Shape::%s for the moment\n\n", type_of_shape.c_str());
            return;
        }

        std::cerr << "\n\nworld_shape(): unknown shape: `" << type_of_shape << "`\n\n\n";
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

        uint batch = 256;
        uint total_jobs = points.size() / batch + 1;
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

        TriangleMesh *mesh;
        checkCudaErrors(cudaMallocManaged((void **)&mesh, sizeof(TriangleMesh)));
        mesh->init(reverse_orientation, gpu_indices, indices.size(), gpu_points, points.size());

        Triangle *triangles;
        checkCudaErrors(
            cudaMallocManaged((void **)&triangles, sizeof(Triangle) * mesh->triangle_num));
        Shape *shapes;
        checkCudaErrors(cudaMallocManaged((void **)&shapes, sizeof(Shape) * mesh->triangle_num));

        {
            uint threads = 1024;
            uint blocks = divide_and_ceil(mesh->triangle_num, threads);
            GPU::init_triangles_from_mesh<<<blocks, threads>>>(triangles, mesh);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            GPU::build_shapes<<<blocks, threads>>>(shapes, triangles, mesh->triangle_num);

            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            for (int idx = 0; idx < mesh->triangle_num; idx++) {
                gpu_primitives.push_back(&shapes[idx]);
            }
        }

        gpu_dynamic_pointers.push_back(gpu_indices);
        gpu_dynamic_pointers.push_back(gpu_points);
        gpu_dynamic_pointers.push_back(gpu_uv);
        gpu_dynamic_pointers.push_back(mesh);
        gpu_dynamic_pointers.push_back(triangles);
        gpu_dynamic_pointers.push_back(shapes);
    }

    void parse_tokens(const std::vector<Token> &tokens) {
        const Token &first_token = tokens[0];

        switch (first_token.type) {
        case TokenType::AttributeBegin: {
            pushed_graphics_state.push(graphics_state);
            return;
        }

        case TokenType::AttributeEnd: {
            graphics_state = pushed_graphics_state.top();
            pushed_graphics_state.pop();
            return;
        }

        case TokenType::WorldBegin: {
            option_film();
            option_filter();
            option_camera();
            option_sampler();
            option_integrator();

            graphics_state.current_transform = Transform::identity();
            named_coordinate_systems["world"] = graphics_state.current_transform;

            return;
        }

        case TokenType::Keyword: {
            const auto keyword = first_token.values[0];

            if (keyword == "Camera") {
                camera_tokens = tokens;
                return;
            }

            if (keyword == "Film") {
                film_tokens = tokens;
                return;
            }

            if (keyword == "Include") {
                auto included_file = tokens[1].values[0];
                auto full_path = root.empty() ? included_file : root + "/" + included_file;
                parse_file(full_path);
                return;
            }

            if (keyword == "Integrator") {
                integrator_name = tokens[1].values[0];
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

            if (keyword == "AreaLightSource" || keyword == "LightSource" || keyword == "Material" ||
                keyword == "MakeNamedMaterial" || keyword == "NamedMaterial" ||
                keyword == "ReverseOrientation" || keyword == "Texture") {
                std::cout << "parse_tokens::Keyword `" << keyword << "` ignored for the moment\n";
                return;
            }

            std::cerr << "\nERROR: parse_tokens::Keyword `" << keyword << "` not implemented\n\n";
            throw std::runtime_error("parse_tokens(): unknown keyword");
        }

        default: {
            std::cout << "Builder::parse_tokens(): unknown token type: " << first_token.type
                      << "\n";
            throw std::runtime_error("parse_tokens() fail");
        }
        }
    }

    void parse_file(const std::string &_filename) {
        const auto all_tokens = parse_pbrt_into_token(_filename);
        const auto range_of_tokens = SceneBuilder::group_tokens(all_tokens);

        for (uint range_idx = 0; range_idx < range_of_tokens.size() - 1; ++range_idx) {
            auto current_tokens = std::vector(all_tokens.begin() + range_of_tokens[range_idx],
                                              all_tokens.begin() + range_of_tokens[range_idx + 1]);

            parse_tokens(current_tokens);
        }
    }

    void preprocess() {
        renderer->bvh->build_bvh(gpu_dynamic_pointers, gpu_primitives);
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

        GPU::parallel_render<<<blocks, threads>>>(renderer, samples_per_pixel.value());

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

        GPU::copy_gpu_pixels_to_rgb<<<blocks, threads>>>(renderer, output_rgb);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        GPU::writer_to_file(output_filename, output_rgb, film_resolution.value());

        checkCudaErrors(cudaFree(output_rgb));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        std::cout << "image saved to `" << output_filename << "`\n";
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
