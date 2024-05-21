#pragma once

#include <chrono>
#include <filesystem>
#include <map>
#include <set>
#include <stack>

#include "pbrt/base/light.h"
#include "pbrt/base/material.h"
#include "pbrt/base/primitive.h"

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/transform.h"

#include "pbrt/filters/box.h"

#include "pbrt/gpu/renderer.h"

#include "pbrt/integrators/ambient_occlusion.h"
#include "pbrt/integrators/surface_normal.h"
#include "pbrt/integrators/random_walk.h"
#include "pbrt/integrators/simple_path.h"

#include "pbrt/lights/diffuse_area_light.h"

#include "pbrt/light_samplers/uniform_light_sampler.h"

#include "pbrt/materials/diffuse_material.h"
#include "pbrt/materials/dielectric_material.h"

#include "pbrt/primitives/simple_primitives.h"
#include "pbrt/primitives/geometric_primitive.h"

#include "pbrt/spectrum_util/rgb_to_spectrum_data.h"
#include "pbrt/spectra/const_spectrum.h"

#include "pbrt/scene/command_line_option.h"
#include "pbrt/scene/parser.h"
#include "pbrt/scene/parameter_dict.h"
#include "pbrt/shapes/loop_subdivide.h"
#include "pbrt/shapes/tri_quad_mesh.h"

#include "pbrt/textures/spectrum_constant_texture.h"
#include "pbrt/textures/spectrum_image_texture.h"
#include "pbrt/textures/spectrum_scale_texture.h"

#include "pbrt/util/std_container.h"

namespace {
std::string get_dirname(const std::string &full_path) {
    const size_t last_slash_idx = full_path.rfind('/');
    if (std::string::npos != last_slash_idx) {
        return full_path.substr(0, last_slash_idx);
    }

    throw std::runtime_error("get_dirname() fails");
}
} // namespace

struct AreaLightEntity {
    std::string name;
    ParameterDict parameters;

    AreaLightEntity(const std::string &_name, const ParameterDict &_parameters)
        : name(_name), parameters(_parameters) {}
};

class GraphicsState {
  public:
    GraphicsState() : current_transform(Transform::identity()), reverse_orientation(false) {}

    Transform current_transform;
    bool reverse_orientation = false;
    Material *current_material = nullptr;

    std::optional<AreaLightEntity> area_light_entity;
};

struct PreComputedSpectrum {
    explicit PreComputedSpectrum(ThreadPool &thread_pool) {
        auto start = std::chrono::system_clock::now();

        for (uint idx = 0; idx < 3; idx++) {
            CHECK_CUDA_ERROR(
                cudaMallocManaged(&(dense_cie_xyz[idx]), sizeof(DenselySampledSpectrum)));
        }
        dense_cie_xyz[0]->init_from_pls_lambdas_values(CIE_LAMBDA_CPU, CIE_X_VALUE_CPU,
                                                       NUM_CIE_SAMPLES);
        dense_cie_xyz[1]->init_from_pls_lambdas_values(CIE_LAMBDA_CPU, CIE_Y_VALUE_CPU,
                                                       NUM_CIE_SAMPLES);
        dense_cie_xyz[2]->init_from_pls_lambdas_values(CIE_LAMBDA_CPU, CIE_Z_VALUE_CPU,
                                                       NUM_CIE_SAMPLES);

        for (uint idx = 0; idx < 3; idx++) {
            Spectrum *_spectrum;
            CHECK_CUDA_ERROR(cudaMallocManaged(&_spectrum, sizeof(Spectrum)));
            _spectrum->init(dense_cie_xyz[idx]);
            cie_xyz[idx] = _spectrum;
        }

        CHECK_CUDA_ERROR(cudaMallocManaged(&dense_illum_d65, sizeof(DenselySampledSpectrum)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&illum_d65, sizeof(Spectrum)));

        dense_illum_d65->init_from_pls_interleaved_samples(
            CIE_Illum_D6500, sizeof(CIE_Illum_D6500) / sizeof(CIE_Illum_D6500[0]), true,
            cie_xyz[1]);
        illum_d65->init(dense_illum_d65);

        CHECK_CUDA_ERROR(cudaMallocManaged(&rgb_to_spectrum_table,
                                           sizeof(RGBtoSpectrumData::RGBtoSpectrumTable)));
        rgb_to_spectrum_table->init("sRGB", thread_pool);

        const std::chrono::duration<FloatType> duration{std::chrono::system_clock::now() - start};
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
            CHECK_CUDA_ERROR(cudaFree(ptr));
        }

        for (uint idx = 0; idx < 3; idx++) {
            CHECK_CUDA_ERROR(cudaFree(dense_cie_xyz[idx]));
            CHECK_CUDA_ERROR(cudaFree((void *)cie_xyz[idx]));
        }

        CHECK_CUDA_ERROR(cudaGetLastError());
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

    std::set<std::string> ignored_keywords;

    [[nodiscard]] std::string get_file_full_path(const std::string &relative_path) const {
        return root.empty() ? relative_path : root + "/" + relative_path;
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
    std::vector<const Light *> gpu_lights;

    GraphicsState graphics_state;
    std::stack<GraphicsState> pushed_graphics_state;
    std::map<std::string, Transform> named_coordinate_systems;
    Transform render_from_world;

    std::map<std::string, const SpectrumTexture *> named_spectrum_texture;

    explicit SceneBuilder(const CommandLineOption &command_line_option)
        : samples_per_pixel(command_line_option.samples_per_pixel),
          pre_computed_spectrum(PreComputedSpectrum(thread_pool)) {

        GPU::GlobalVariable *global_variables;
        CHECK_CUDA_ERROR(cudaMallocManaged(&global_variables, sizeof(GPU::GlobalVariable)));

        CHECK_CUDA_ERROR(
            cudaMallocManaged(&(global_variables->rgb_color_space), sizeof(RGBColorSpace)));

        global_variables->init(pre_computed_spectrum.cie_xyz, pre_computed_spectrum.illum_d65,
                               pre_computed_spectrum.rgb_to_spectrum_table,
                               RGBtoSpectrumData::Gamut::sRGB);

        CHECK_CUDA_ERROR(cudaMallocManaged(&renderer, sizeof(GPU::Renderer)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&(renderer->bvh), sizeof(HLBVH)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&(renderer->camera), sizeof(Camera)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&(renderer->film), sizeof(Film)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&(renderer->filter), sizeof(Filter)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&(renderer->integrator), sizeof(Integrator)));

        renderer->global_variables = global_variables;

        for (auto ptr : std::vector<void *>({
                 global_variables,
                 global_variables->rgb_color_space,
                 renderer,
                 renderer->bvh,
                 renderer->camera,
                 renderer->film,
                 renderer->filter,
                 renderer->integrator,
             })) {
            gpu_dynamic_pointers.push_back(ptr);
        }

        ConstantSpectrum *constant_grey;
        Spectrum *spectrum_grey;
        SpectrumConstantTexture *texture_grey;
        SpectrumTexture *texture;
        DiffuseMaterial *default_diffuse_material;
        Material *default_material;

        CHECK_CUDA_ERROR(cudaMallocManaged(&constant_grey, sizeof(ConstantSpectrum)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_grey, sizeof(Spectrum)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&texture_grey, sizeof(SpectrumConstantTexture)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&texture, sizeof(SpectrumTexture)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&default_diffuse_material, sizeof(DiffuseMaterial)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&default_material, sizeof(Material)));

        constant_grey->init(0.5);
        spectrum_grey->init(constant_grey);
        texture_grey->init(spectrum_grey);
        texture->init(texture_grey);
        default_diffuse_material->init_reflectance(texture);
        default_material->init(default_diffuse_material);

        graphics_state.current_material = default_material;

        for (auto ptr : std::vector<void *>({
                 constant_grey,
                 spectrum_grey,
                 texture_grey,
                 texture,
                 default_diffuse_material,
                 default_material,
             })) {
            gpu_dynamic_pointers.push_back(ptr);
        }
    }

    ~SceneBuilder() {
        for (auto ptr : gpu_dynamic_pointers) {
            CHECK_CUDA_ERROR(cudaFree(ptr));
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
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

    void build_camera() {
        auto parameters =
            ParameterDict(sub_vector(camera_tokens, 2), named_spectrum_texture, this->root);
        const auto camera_type = camera_tokens[1].values[0];
        if (camera_type == "perspective") {
            auto camera_from_world = graphics_state.current_transform;
            auto world_from_camera = camera_from_world.inverse();

            named_coordinate_systems["camera"] = world_from_camera;

            auto camera_transform = CameraTransform(
                world_from_camera, RenderingCoordinateSystem::CameraWorldCoordSystem);

            render_from_world = camera_transform.render_from_world;

            FloatType fov = parameters.get_float("fov", 90);

            PerspectiveCamera *perspective_camera;
            CHECK_CUDA_ERROR(cudaMallocManaged(&perspective_camera, sizeof(PerspectiveCamera)));
            gpu_dynamic_pointers.push_back(perspective_camera);

            perspective_camera->init(film_resolution.value(), camera_transform, fov, 0.0);
            renderer->camera->init(perspective_camera);
            return;
        }

        std::cerr << "Camera type `" << camera_type << "` not implemented\n";
        throw std::runtime_error("camera type not implemented");
    }

    void build_filter() {
        BoxFilter *box_filter;
        CHECK_CUDA_ERROR(cudaMallocManaged(&box_filter, sizeof(BoxFilter)));
        gpu_dynamic_pointers.push_back(box_filter);

        box_filter->init(0.5);
        renderer->filter->init(box_filter);
    }

    void build_film() {
        auto parameters =
            ParameterDict(sub_vector(film_tokens, 2), named_spectrum_texture, this->root);

        auto _resolution_x = parameters.get_integer("xresolution")[0];
        auto _resolution_y = parameters.get_integer("yresolution")[0];

        film_resolution = Point2i(_resolution_x, _resolution_y);
        output_filename = parameters.get_string("filename", std::nullopt);

        if (std::filesystem::path p(output_filename); p.extension() != ".png") {
            printf("output filename extension: only PNG is supported for the moment\n");
            output_filename = p.replace_extension(".png").filename();
        }

        DenselySampledSpectrum *_d_illum_dense;
        Spectrum *d_illum;

        CHECK_CUDA_ERROR(cudaMallocManaged(&_d_illum_dense, sizeof(DenselySampledSpectrum)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&d_illum, sizeof(Spectrum)));

        gpu_dynamic_pointers.push_back(d_illum);
        gpu_dynamic_pointers.push_back(_d_illum_dense);

        FloatType iso = 100;
        FloatType white_balance_val = 0.0;
        FloatType exposure_time = 1.0;
        FloatType imaging_ratio = exposure_time * iso / 100.0;

        _d_illum_dense->init_cie_d(white_balance_val == 0.0 ? 6500.0 : white_balance_val, CIE_S0,
                                   CIE_S1, CIE_S2, CIE_S_lambda);

        d_illum->init(_d_illum_dense);

        const Spectrum *cie_xyz[3];
        renderer->global_variables->get_cie_xyz(cie_xyz);

        renderer->sensor.init_cie_1931(cie_xyz, renderer->global_variables->rgb_color_space,
                                       white_balance_val == 0 ? nullptr : d_illum, imaging_ratio);

        Pixel *gpu_pixels;
        CHECK_CUDA_ERROR(cudaMallocManaged(&gpu_pixels, sizeof(Pixel) * film_resolution->x *
                                                            film_resolution->y));
        gpu_dynamic_pointers.push_back(gpu_pixels);

        {
            uint threads = 1024;
            uint blocks = divide_and_ceil(uint(film_resolution->x * film_resolution->y), threads);

            GPU::init_pixels<<<blocks, threads>>>(gpu_pixels, film_resolution.value());
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }

        RGBFilm *rgb_film;
        CHECK_CUDA_ERROR(cudaMallocManaged(&rgb_film, sizeof(RGBFilm)));
        gpu_dynamic_pointers.push_back(rgb_film);

        rgb_film->init(gpu_pixels, &(renderer->sensor), film_resolution.value(),
                       renderer->global_variables->rgb_color_space);

        renderer->film->init(rgb_film);
    }

    void build_sampler() {
        // TODO: sampler is not parsed, only pixelsamples read
        const auto parameters =
            ParameterDict(sub_vector(sampler_tokens, 2), named_spectrum_texture, this->root);
        auto samples_from_parameters = parameters.get_integer("pixelsamples");

        if (!samples_per_pixel) {
            if (!samples_from_parameters.empty()) {
                samples_per_pixel = samples_from_parameters[0];
            } else {
                samples_per_pixel = 4;
                // default samples per pixel
            }
        }

        uint total_pixel_num = film_resolution->x * film_resolution->y;

        IndependentSampler *independent_samplers;

        CHECK_CUDA_ERROR(
            cudaMallocManaged(&(renderer->samplers), sizeof(Sampler) * total_pixel_num));
        CHECK_CUDA_ERROR(
            cudaMallocManaged(&independent_samplers, sizeof(IndependentSampler) * total_pixel_num));

        {
            uint threads = 1024;
            uint blocks = divide_and_ceil(total_pixel_num, threads);

            GPU::init_independent_samplers<<<blocks, threads>>>(
                independent_samplers, samples_per_pixel.value(), total_pixel_num);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            GPU::init_samplers<<<blocks, threads>>>(renderer->samplers, independent_samplers,
                                                    total_pixel_num);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }

        gpu_dynamic_pointers.push_back(renderer->samplers);
        gpu_dynamic_pointers.push_back(independent_samplers);
    }

    void build_integrator() {
        IntegratorBase *integrator_base;
        CHECK_CUDA_ERROR(cudaMallocManaged(&integrator_base, sizeof(IntegratorBase)));

        const Light **light_array;
        CHECK_CUDA_ERROR(cudaMallocManaged(&light_array, sizeof(Light *) * gpu_lights.size()));
        CHECK_CUDA_ERROR(cudaMemcpy(light_array, gpu_lights.data(),
                                    sizeof(Light *) * gpu_lights.size(), cudaMemcpyHostToDevice));

        UniformLightSampler *uniform_light_sampler;
        CHECK_CUDA_ERROR(cudaMallocManaged(&uniform_light_sampler, sizeof(UniformLightSampler)));
        uniform_light_sampler->lights = light_array;
        uniform_light_sampler->light_num = gpu_lights.size();

        integrator_base->bvh = renderer->bvh;
        integrator_base->camera = renderer->camera;
        integrator_base->lights = light_array;
        integrator_base->light_num = gpu_lights.size();

        integrator_base->uniform_light_sampler = uniform_light_sampler;

        for (auto ptr : std::vector<void *>({
                 integrator_base,
                 light_array,
                 uniform_light_sampler,
             })) {
            gpu_dynamic_pointers.push_back(ptr);
        }

        if (!integrator_name.has_value()) {
            printf("Integrator not set, changed to AmbientOcclusion\n");
            integrator_name = "ambientocclusion";
        }

        if (integrator_name == "volpath") {
            printf("Integrator `%s` not implemented, changed to AmbientOcclusion\n",
                   integrator_name->c_str());
            integrator_name = "ambientocclusion";
        }

        if (integrator_name == "ambientocclusion") {
            auto illuminant_spectrum = renderer->global_variables->rgb_color_space->illuminant;

            const Spectrum *cie_xyz[3];
            renderer->global_variables->get_cie_xyz(cie_xyz);
            const auto cie_y = cie_xyz[1];
            auto illuminant_scale = 1.0 / illuminant_spectrum->to_photometric(cie_y);

            AmbientOcclusionIntegrator *ambient_occlusion_integrator;
            CHECK_CUDA_ERROR(cudaMallocManaged(&ambient_occlusion_integrator,
                                               sizeof(AmbientOcclusionIntegrator)));
            gpu_dynamic_pointers.push_back(ambient_occlusion_integrator);

            ambient_occlusion_integrator->init(integrator_base, illuminant_spectrum,
                                               illuminant_scale);
            renderer->integrator->init(ambient_occlusion_integrator);
            return;
        }

        if (integrator_name == "surfacenormal") {
            SurfaceNormalIntegrator *surface_normal_integrator;
            CHECK_CUDA_ERROR(
                cudaMallocManaged(&surface_normal_integrator, sizeof(SurfaceNormalIntegrator)));
            gpu_dynamic_pointers.push_back(surface_normal_integrator);

            surface_normal_integrator->init(integrator_base,
                                            renderer->global_variables->rgb_color_space);
            renderer->integrator->init(surface_normal_integrator);
            return;
        }

        if (integrator_name == "randomwalk") {
            RandomWalkIntegrator *random_walk_integrator;
            CHECK_CUDA_ERROR(
                cudaMallocManaged(&random_walk_integrator, sizeof(RandomWalkIntegrator)));
            gpu_dynamic_pointers.push_back(random_walk_integrator);

            random_walk_integrator->init(integrator_base, 5);
            renderer->integrator->init(random_walk_integrator);
            return;
        }

        if (integrator_name == "simplepath") {
            SimplePathIntegrator *simple_path_integrator;
            CHECK_CUDA_ERROR(
                cudaMallocManaged(&simple_path_integrator, sizeof(SimplePathIntegrator)));
            gpu_dynamic_pointers.push_back(simple_path_integrator);

            simple_path_integrator->init(integrator_base, 5);
            renderer->integrator->init(simple_path_integrator);

            return;
        }

        const std::string error =
            "parse_tokens(): unknown Integrator name: `" + integrator_name.value() + "`";
        throw std::runtime_error(error.c_str());
    }

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

        graphics_state.current_transform =
            graphics_state.current_transform * Transform::lookat(position, look, up);
    }

    void parse_material(const std::vector<Token> &tokens) {
        if (tokens[0] != Token(TokenType::Keyword, "Material")) {
            throw std::runtime_error("expect Keyword(Material)");
        }

        const auto parameters =
            ParameterDict(sub_vector(tokens, 2), named_spectrum_texture, this->root);
        auto type_of_material = tokens[1].values[0];

        if (type_of_material == "diffuse") {
            DiffuseMaterial *diffuse_material;
            Material *material;
            CHECK_CUDA_ERROR(cudaMallocManaged(&diffuse_material, sizeof(DiffuseMaterial)));
            CHECK_CUDA_ERROR(cudaMallocManaged(&material, sizeof(Material)));

            diffuse_material->init(renderer->global_variables->rgb_color_space, parameters,
                                   gpu_dynamic_pointers);
            material->init(diffuse_material);

            graphics_state.current_material = material;

            gpu_dynamic_pointers.push_back(diffuse_material);
            gpu_dynamic_pointers.push_back(material);
            return;
        }

        if (type_of_material == "dielectric") {
            DielectricMaterial *dielectric_material;
            Material *material;

            CHECK_CUDA_ERROR(cudaMallocManaged(&dielectric_material, sizeof(DielectricMaterial)));
            CHECK_CUDA_ERROR(cudaMallocManaged(&material, sizeof(Material)));

            dielectric_material->init(parameters, gpu_dynamic_pointers);
            material->init(dielectric_material);
            graphics_state.current_material = material;

            gpu_dynamic_pointers.push_back(dielectric_material);
            gpu_dynamic_pointers.push_back(material);
            return;
        }

        const std::string error =
            "parse_material(): unknown Material name: `" + type_of_material + "`";
        throw std::runtime_error(error.c_str());
    }

    void parse_rotate(const std::vector<Token> &tokens) {
        if (tokens[0] != Token(TokenType::Keyword, "Rotate")) {
            throw std::runtime_error("expect Keyword(Rotate)");
        }

        std::vector<FloatType> data;
        for (int idx = 1; idx < tokens.size(); idx++) {
            data.push_back(tokens[idx].to_float());
        }

        graphics_state.current_transform = graphics_state.current_transform *
                                           Transform::rotate(data[0], data[1], data[2], data[3]);
    }

    void parse_scale(const std::vector<Token> &tokens) {
        if (tokens[0] != Token(TokenType::Keyword, "Scale")) {
            throw std::runtime_error("expect Keyword(Scale)");
        }

        std::vector<FloatType> data;
        for (int idx = 1; idx < tokens.size(); idx++) {
            data.push_back(tokens[idx].to_float());
        }

        graphics_state.current_transform *= Transform::scale(data[0], data[1], data[2]);
    }

    void world_area_light_source(const std::vector<Token> &tokens) {
        if (tokens[0] != Token(TokenType::Keyword, "AreaLightSource")) {
            throw std::runtime_error("expect Keyword(AreaLightSource)");
        }

        if (tokens[1] != Token(TokenType::String, "diffuse")) {
            throw std::runtime_error(
                "world_area_light_source: only `diffuse` supported at the moment");
        }

        graphics_state.area_light_entity =
            AreaLightEntity(tokens[1].values[0], ParameterDict(sub_vector(tokens, 2),
                                                               named_spectrum_texture, this->root));
    }

    void parse_shape(const std::vector<Token> &tokens) {
        if (tokens[0] != Token(TokenType::Keyword, "Shape")) {
            throw std::runtime_error("expect Keyword(Shape)");
        }

        const auto parameters =
            ParameterDict(sub_vector(tokens, 2), named_spectrum_texture, this->root);

        auto type_of_shape = tokens[1].values[0];
        if (type_of_shape == "trianglemesh") {
            auto uv = parameters.get_point2("uv");
            auto indices = parameters.get_integer("indices");
            auto points = parameters.get_point3("P");

            add_triangle_mesh(points, indices, uv);

            return;
        }

        if (type_of_shape == "plymesh") {
            auto file_path = get_file_full_path(parameters.get_string("filename", std::nullopt));
            auto ply_mesh = TriQuadMesh::read_ply(file_path);

            // TODO: displacement texture is not implemented here
            if (!ply_mesh.triIndices.empty()) {
                add_triangle_mesh(ply_mesh.p, ply_mesh.triIndices, ply_mesh.uv);
            }

            if (!ply_mesh.quadIndices.empty()) {
                printf("\n%s(): Shape::plymesh.quadIndices not implemented\n", __func__);
                REPORT_FATAL_ERROR();
            }

            return;
        }

        if (type_of_shape == "loopsubdiv") {
            auto levels = parameters.get_integer("levels")[0];
            auto indices = parameters.get_integer("indices");
            auto points = parameters.get_point3("P");

            const auto loop_subdivide_data = LoopSubdivide(levels, indices, points);

            add_triangle_mesh(loop_subdivide_data.p_limit, loop_subdivide_data.vertex_indices,
                              std::vector<Point2f>());
            return;
        }

        if (type_of_shape == "disk" || type_of_shape == "sphere") {
            printf("\nignore Shape::%s for the moment\n\n", type_of_shape.c_str());
            return;
        }

        std::cout << "\n" << __func__ << "(): unknown shape: `" << type_of_shape << "`\n\n\n";
        REPORT_FATAL_ERROR();
    }

    void parse_texture(const std::vector<Token> &tokens) {
        auto texture_name = tokens[1].values[0];
        auto color_type = tokens[2].values[0];
        auto texture_type = tokens[3].values[0];

        if (color_type == "spectrum") {
            auto parameters =
                ParameterDict(sub_vector(tokens, 4), named_spectrum_texture, this->root);
            if (texture_type == "imagemap") {
                SpectrumImageTexture *image_texture;
                SpectrumTexture *spectrum_texture;

                CHECK_CUDA_ERROR(cudaMallocManaged(&image_texture, sizeof(SpectrumImageTexture)));
                CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_texture, sizeof(SpectrumTexture)));

                image_texture->new_init(parameters, renderer->global_variables->rgb_color_space,
                                        gpu_dynamic_pointers);
                spectrum_texture->init(image_texture);

                /*
                auto filename = parameters.get_string("filename", std::nullopt);
                auto absolute_path = this->root + "/" + filename;

                SpectrumImageTexture *image_texture;
                SpectrumTexture *spectrum_texture;
                CHECK_CUDA_ERROR(cudaMallocManaged(&image_texture, sizeof(SpectrumImageTexture)));
                CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_texture, sizeof(SpectrumTexture)));

                image_texture->init(parameters, get_render_from_object(), absolute_path,
                                    renderer->global_variables->rgb_color_space,
                                    gpu_dynamic_pointers);
                spectrum_texture->init(image_texture);
                */
                named_spectrum_texture[texture_name] = spectrum_texture;

                gpu_dynamic_pointers.push_back(image_texture);
                gpu_dynamic_pointers.push_back(spectrum_texture);

                return;
            }

            if (texture_type == "scale") {
                auto base_texture = parameters.get_spectrum_texture("tex");
                auto scale = parameters.get_float("scale", std::optional(1.0));

                SpectrumScaleTexture *scale_texture;
                SpectrumTexture *spectrum_texture;

                CHECK_CUDA_ERROR(cudaMallocManaged(&scale_texture, sizeof(SpectrumScaleTexture)));
                CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_texture, sizeof(SpectrumTexture)));

                scale_texture->init(base_texture, scale);
                spectrum_texture->init(scale_texture);

                named_spectrum_texture[texture_name] = spectrum_texture;

                gpu_dynamic_pointers.push_back(scale_texture);
                gpu_dynamic_pointers.push_back(spectrum_texture);

                return;
            }

            REPORT_FATAL_ERROR();
        }

        REPORT_FATAL_ERROR();
    }

    void parse_transform(const std::vector<Token> &tokens) {
        if (tokens[0] != Token(TokenType::Keyword, "Transform")) {
            std::cout << "\nexpect Keyword(Transform)\n";
            REPORT_FATAL_ERROR();
        }

        std::vector<FloatType> data(16);
        for (uint idx = 0; idx < tokens[1].values.size(); idx++) {
            data[idx] = stod(tokens[1].values[idx]);
        }

        FloatType transform_data[4][4];
        for (uint y = 0; y < 4; y++) {
            for (uint x = 0; x < 4; x++) {
                transform_data[y][x] = data[y * 4 + x];
            }
        }

        auto transform_matrix = SquareMatrix<4>(transform_data);

        graphics_state.current_transform = transform_matrix.transpose();
    }

    void parse_translate(const std::vector<Token> &tokens) {
        std::vector<FloatType> data;
        for (int idx = 1; idx < tokens.size(); idx++) {
            data.push_back(tokens[idx].to_float());
        }

        assert(data.size() == 16);

        graphics_state.current_transform *= Transform::translate(data[0], data[1], data[2]);
    }

    void add_triangle_mesh(const std::vector<Point3f> &points, const std::vector<int> &indices,
                           const std::vector<Point2f> &uv) {
        const auto render_from_object = get_render_from_object();
        const bool reverse_orientation = graphics_state.reverse_orientation;

        Point3f *gpu_points;
        CHECK_CUDA_ERROR(cudaMallocManaged(&gpu_points, sizeof(Point3f) * points.size()));
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_points, points.data(), sizeof(Point3f) * points.size(),
                                    cudaMemcpyHostToDevice));

        uint batch = 256;
        uint total_jobs = points.size() / batch + 1;
        GPU::apply_transform<<<total_jobs, batch>>>(gpu_points, render_from_object, points.size());
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        int *gpu_indices;
        CHECK_CUDA_ERROR(cudaMallocManaged(&gpu_indices, sizeof(int) * indices.size()));
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_indices, indices.data(), sizeof(int) * indices.size(),
                                    cudaMemcpyHostToDevice));

        Point2f *gpu_uv = nullptr;
        if (!uv.empty()) {
            CHECK_CUDA_ERROR(cudaMallocManaged(&gpu_uv, sizeof(Point2f) * uv.size()));
            CHECK_CUDA_ERROR(
                cudaMemcpy(gpu_uv, uv.data(), sizeof(Point2f) * uv.size(), cudaMemcpyHostToDevice));
            gpu_dynamic_pointers.push_back(gpu_uv);
        }

        TriangleMesh *mesh;
        CHECK_CUDA_ERROR(cudaMallocManaged(&mesh, sizeof(TriangleMesh)));
        mesh->init(reverse_orientation, gpu_indices, indices.size(), gpu_points, gpu_uv);

        uint num_triangles = mesh->triangles_num;
        Triangle *triangles;
        CHECK_CUDA_ERROR(cudaMallocManaged(&triangles, sizeof(Triangle) * num_triangles));
        Shape *shapes;
        CHECK_CUDA_ERROR(cudaMallocManaged(&shapes, sizeof(Shape) * num_triangles));

        {
            uint threads = 1024;
            uint blocks = divide_and_ceil(num_triangles, threads);

            GPU::init_triangles_from_mesh<<<blocks, threads>>>(triangles, mesh);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            GPU::init_shapes<<<blocks, threads>>>(shapes, triangles, num_triangles);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            if (!graphics_state.area_light_entity) {
                SimplePrimitive *simple_primitives;
                CHECK_CUDA_ERROR(
                    cudaMallocManaged(&simple_primitives, sizeof(SimplePrimitive) * num_triangles));
                Primitive *primitives;
                CHECK_CUDA_ERROR(cudaMallocManaged(&primitives, sizeof(Primitive) * num_triangles));

                GPU::init_simple_primitives<<<blocks, threads>>>(
                    simple_primitives, shapes, graphics_state.current_material, num_triangles);
                CHECK_CUDA_ERROR(cudaGetLastError());
                CHECK_CUDA_ERROR(cudaDeviceSynchronize());

                GPU::init_primitives<<<blocks, threads>>>(primitives, simple_primitives,
                                                          num_triangles);
                CHECK_CUDA_ERROR(cudaGetLastError());
                CHECK_CUDA_ERROR(cudaDeviceSynchronize());

                gpu_primitives.reserve(gpu_primitives.size() + num_triangles);
                for (uint idx = 0; idx < num_triangles; idx++) {
                    gpu_primitives.push_back(&primitives[idx]);
                }

                gpu_dynamic_pointers.push_back(simple_primitives);
                gpu_dynamic_pointers.push_back(primitives);

            } else {
                // build light
                DiffuseAreaLight *diffuse_area_lights;
                CHECK_CUDA_ERROR(cudaMallocManaged(&diffuse_area_lights,
                                                   sizeof(DiffuseAreaLight) * num_triangles));

                Light *lights;
                CHECK_CUDA_ERROR(cudaMallocManaged(&lights, sizeof(Light) * num_triangles));

                GeometricPrimitive *geometric_primitives;
                CHECK_CUDA_ERROR(cudaMallocManaged(&geometric_primitives,
                                                   sizeof(GeometricPrimitive) * num_triangles));

                for (uint idx = 0; idx < num_triangles; idx++) {
                    diffuse_area_lights[idx].init(render_from_object,
                                                  graphics_state.area_light_entity->parameters,
                                                  &shapes[idx], renderer->global_variables);
                }

                GPU::init_lights<<<blocks, threads>>>(lights, diffuse_area_lights, num_triangles);
                CHECK_CUDA_ERROR(cudaGetLastError());
                CHECK_CUDA_ERROR(cudaDeviceSynchronize());

                GPU::init_geometric_primitives<<<blocks, threads>>>(geometric_primitives, shapes,
                                                                    graphics_state.current_material,
                                                                    lights, num_triangles);
                CHECK_CUDA_ERROR(cudaGetLastError());
                CHECK_CUDA_ERROR(cudaDeviceSynchronize());

                Primitive *primitives;
                CHECK_CUDA_ERROR(cudaMallocManaged(&primitives, sizeof(Primitive) * num_triangles));

                GPU::init_primitives<<<blocks, threads>>>(primitives, geometric_primitives,
                                                          num_triangles);
                CHECK_CUDA_ERROR(cudaGetLastError());
                CHECK_CUDA_ERROR(cudaDeviceSynchronize());

                gpu_primitives.reserve(gpu_primitives.size() + num_triangles);
                for (uint idx = 0; idx < num_triangles; idx++) {
                    gpu_primitives.push_back(&primitives[idx]);
                }

                gpu_lights.reserve(gpu_lights.size() + num_triangles);
                for (uint idx = 0; idx < num_triangles; idx++) {
                    gpu_lights.push_back(&lights[idx]);
                }

                for (auto ptr : std::vector<void *>({
                         diffuse_area_lights,
                         lights,
                         geometric_primitives,
                         primitives,
                     })) {
                    gpu_dynamic_pointers.push_back(ptr);
                }
            }
        }

        for (auto ptr : std::vector<void *>({
                 gpu_indices,
                 gpu_points,
                 mesh,
                 triangles,
                 shapes,
             })) {
            gpu_dynamic_pointers.push_back(ptr);
        }
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
            build_film();
            build_filter();
            build_camera();
            build_sampler();

            graphics_state.current_transform = Transform::identity();
            named_coordinate_systems["world"] = graphics_state.current_transform;

            return;
        }

        case TokenType::Keyword: {
            const auto keyword = first_token.values[0];

            if (keyword == "AreaLightSource") {
                if (should_ignore_material_and_texture()) {
                    return;
                }

                world_area_light_source(tokens);
                return;
            }

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
                parse_file(get_file_full_path(included_file));
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

            if (keyword == "Material") {
                if (should_ignore_material_and_texture()) {
                    printf("ignoring %s\n", keyword.c_str());
                    return;
                }

                parse_material(tokens);
                return;
            }

            if (keyword == "ReverseOrientation") {
                graphics_state.reverse_orientation = !graphics_state.reverse_orientation;
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
                parse_shape(tokens);
                return;
            }

            if (keyword == "Texture") {
                if (should_ignore_material_and_texture()) {
                    printf("ignoring %s\n", keyword.c_str());
                    return;
                }

                parse_texture(tokens);
                return;
            }

            if (keyword == "Transform") {
                parse_transform(tokens);
                return;
            }

            if (keyword == "Translate") {
                parse_translate(tokens);
                return;
            }

            if (keyword == "ConcatTransform" || keyword == "LightSource" ||
                keyword == "MakeNamedMaterial" || keyword == "MakeNamedMedium" ||
                keyword == "MediumInterface" || keyword == "NamedMaterial" ||
                keyword == "ObjectBegin" || keyword == "ObjectEnd" || keyword == "ObjectInstance" ||
                keyword == "PixelFilter") {

                if (ignored_keywords.find(keyword) == ignored_keywords.end()) {
                    std::cout << "parse_tokens::Keyword `" << keyword
                              << "` ignored for the moment\n";
                    ignored_keywords.insert(keyword);
                }

                return;
            }

            printf("\n%s(): `%s` not implemented\n", __func__, keyword.c_str());
            REPORT_FATAL_ERROR();
        }
        }

        std::cout << __func__ << "(): unknown token type: " << first_token << "\n\n";
        REPORT_FATAL_ERROR();
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
        renderer->bvh->build_bvh(thread_pool, gpu_dynamic_pointers, gpu_primitives);
        build_integrator();
    }

    void render() const {
        auto start_rendering = std::chrono::system_clock::now();

        int thread_width = 8;
        int thread_height = 8;

        std::cout << "\n";
        std::cout << "rendering a " << film_resolution->x << "x" << film_resolution->y
                  << " image (samples per pixel: " << samples_per_pixel.value() << ") ";
        std::cout << "in " << thread_width << "x" << thread_height << " blocks.\n" << std::flush;

        dim3 blocks(divide_and_ceil(uint(film_resolution->x), uint(thread_width)),
                    divide_and_ceil(uint(film_resolution->y), uint(thread_height)), 1);
        dim3 threads(thread_width, thread_height, 1);

        GPU::parallel_render<<<blocks, threads>>>(renderer, samples_per_pixel.value());
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        const std::chrono::duration<FloatType> duration_rendering{std::chrono::system_clock::now() -
                                                                  start_rendering};

        std::cout << std::fixed << std::setprecision(1) << "rendering took "
                  << duration_rendering.count() << " seconds.\n"
                  << std::flush;

        renderer->film->write_to_png(output_filename, film_resolution.value());

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        std::cout << "image saved to `" << output_filename << "`\n";
    }

    bool should_ignore_material_and_texture() const {
        return integrator_name == "ambientocclusion" || integrator_name == "surfacenormal";
    }

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
