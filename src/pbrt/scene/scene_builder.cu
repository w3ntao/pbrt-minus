#include "pbrt/scene/scene_builder.h"

#include "pbrt/filters/box.h"

#include "pbrt/integrators/ambient_occlusion.h"
#include "pbrt/integrators/surface_normal.h"
#include "pbrt/integrators/random_walk.h"
#include "pbrt/integrators/simple_path.h"

#include "pbrt/lights/image_infinite_light.h"

#include "pbrt/materials/coated_diffuse_material.h"
#include "pbrt/materials/diffuse_material.h"
#include "pbrt/materials/dielectric_material.h"

#include "pbrt/shapes/loop_subdivide.h"
#include "pbrt/shapes/tri_quad_mesh.h"

#include "pbrt/textures/spectrum_constant_texture.h"
#include "pbrt/textures/spectrum_image_texture.h"
#include "pbrt/textures/spectrum_scale_texture.h"

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

SceneBuilder::SceneBuilder(const CommandLineOption &command_line_option)
    : samples_per_pixel(command_line_option.samples_per_pixel),
      output_filename(command_line_option.output_file),
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

    auto texture = SpectrumTexture::create_constant_float_val_texture(0.5, gpu_dynamic_pointers);
    graphics_state.material = Material::create_diffuse_material(texture, gpu_dynamic_pointers);
}

void SceneBuilder::build_camera() {
    const auto parameters = build_parameter_dictionary(sub_vector(camera_tokens, 2));

    const auto camera_type = camera_tokens[1].values[0];
    if (camera_type == "perspective") {
        auto camera_from_world = graphics_state.transform;
        auto world_from_camera = camera_from_world.inverse();

        named_coordinate_systems["camera"] = world_from_camera;

        auto camera_transform =
            CameraTransform(world_from_camera, RenderingCoordinateSystem::CameraWorldCoordSystem);

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

void SceneBuilder::build_filter() {
    BoxFilter *box_filter;
    CHECK_CUDA_ERROR(cudaMallocManaged(&box_filter, sizeof(BoxFilter)));
    gpu_dynamic_pointers.push_back(box_filter);

    box_filter->init(0.5);
    renderer->filter->init(box_filter);
}

void SceneBuilder::build_film() {
    const auto parameters = build_parameter_dictionary(sub_vector(film_tokens, 2));

    auto _resolution_x = parameters.get_integer("xresolution")[0];
    auto _resolution_y = parameters.get_integer("yresolution")[0];

    film_resolution = Point2i(_resolution_x, _resolution_y);
    if (output_filename.empty()) {
        output_filename = parameters.get_string("filename", std::nullopt);
    }

    if (std::filesystem::path p(output_filename); p.extension() != ".png") {
        printf("output filename extension: only PNG is supported for the moment\n");
        output_filename = p.replace_extension(".png").filename();
    }

    FloatType iso = 100;
    FloatType white_balance_val = 0.0;
    FloatType exposure_time = 1.0;
    FloatType imaging_ratio = exposure_time * iso / 100.0;

    auto d_illum =
        Spectrum::create_cie_d(white_balance_val == 0.0 ? 6500.0 : white_balance_val, CIE_S0,
                               CIE_S1, CIE_S2, CIE_S_lambda, gpu_dynamic_pointers);

    const Spectrum *cie_xyz[3];
    renderer->global_variables->get_cie_xyz(cie_xyz);

    renderer->sensor.init_cie_1931(cie_xyz, renderer->global_variables->rgb_color_space,
                                   white_balance_val == 0 ? nullptr : d_illum, imaging_ratio);

    Pixel *gpu_pixels;
    CHECK_CUDA_ERROR(
        cudaMallocManaged(&gpu_pixels, sizeof(Pixel) * film_resolution->x * film_resolution->y));
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

void SceneBuilder::build_sampler() {
    // TODO: sampler is not parsed, only pixelsamples read
    const auto parameters = build_parameter_dictionary(sub_vector(sampler_tokens, 2));

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

    CHECK_CUDA_ERROR(cudaMallocManaged(&(renderer->samplers), sizeof(Sampler) * total_pixel_num));
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

const Material *SceneBuilder::create_material(const std::string &type_of_material,
                                              const ParameterDict &parameters) {
    if (type_of_material == "conductor") {
        return Material::create_conductor_material(parameters, gpu_dynamic_pointers);
    }

    if (type_of_material == "coateddiffuse") {
        return Material::create_coated_diffuse_material(parameters, gpu_dynamic_pointers);
    }

    if (type_of_material == "dielectric") {
        DielectricMaterial *dielectric_material;
        Material *material;

        CHECK_CUDA_ERROR(cudaMallocManaged(&dielectric_material, sizeof(DielectricMaterial)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&material, sizeof(Material)));

        gpu_dynamic_pointers.push_back(dielectric_material);
        gpu_dynamic_pointers.push_back(material);

        dielectric_material->init(parameters, gpu_dynamic_pointers);
        material->init(dielectric_material);

        return material;
    }

    if (type_of_material == "diffuse") {
        DiffuseMaterial *diffuse_material;
        Material *material;
        CHECK_CUDA_ERROR(cudaMallocManaged(&diffuse_material, sizeof(DiffuseMaterial)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&material, sizeof(Material)));

        gpu_dynamic_pointers.push_back(diffuse_material);
        gpu_dynamic_pointers.push_back(material);

        diffuse_material->init(parameters, gpu_dynamic_pointers);
        material->init(diffuse_material);

        return material;
    }

    printf("%s(): unknown Material name: `%s`", __func__, type_of_material.c_str());
    REPORT_FATAL_ERROR();
    return nullptr;
}

void SceneBuilder::build_integrator() {
    IntegratorBase *integrator_base;
    CHECK_CUDA_ERROR(cudaMallocManaged(&integrator_base, sizeof(IntegratorBase)));

    const Light **light_array;
    CHECK_CUDA_ERROR(cudaMallocManaged(&light_array, sizeof(Light *) * gpu_lights.size()));
    CHECK_CUDA_ERROR(cudaMemcpy(light_array, gpu_lights.data(), sizeof(Light *) * gpu_lights.size(),
                                cudaMemcpyHostToDevice));

    UniformLightSampler *uniform_light_sampler;
    CHECK_CUDA_ERROR(cudaMallocManaged(&uniform_light_sampler, sizeof(UniformLightSampler)));
    uniform_light_sampler->lights = light_array;
    uniform_light_sampler->light_num = gpu_lights.size();

    std::vector<const Light *> infinite_lights;
    for (auto light : gpu_lights) {
        if (light->get_light_type() == LightType::infinite) {
            infinite_lights.push_back(light);
        }
    }

    const Light **gpu_infinite_lights;
    CHECK_CUDA_ERROR(
        cudaMallocManaged(&gpu_infinite_lights, sizeof(Light *) * infinite_lights.size()));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_infinite_lights, infinite_lights.data(),
                                sizeof(Light *) * infinite_lights.size(), cudaMemcpyHostToDevice));

    integrator_base->bvh = renderer->bvh;
    integrator_base->camera = renderer->camera;

    integrator_base->lights = light_array;
    integrator_base->light_num = gpu_lights.size();

    integrator_base->uniform_light_sampler = uniform_light_sampler;

    integrator_base->infinite_lights = gpu_infinite_lights;
    integrator_base->infinite_light_num = infinite_lights.size();

    for (auto ptr : std::vector<void *>({
             integrator_base,
             light_array,
             uniform_light_sampler,
             gpu_infinite_lights,
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
        CHECK_CUDA_ERROR(
            cudaMallocManaged(&ambient_occlusion_integrator, sizeof(AmbientOcclusionIntegrator)));
        gpu_dynamic_pointers.push_back(ambient_occlusion_integrator);

        ambient_occlusion_integrator->init(integrator_base, illuminant_spectrum, illuminant_scale);
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
        CHECK_CUDA_ERROR(cudaMallocManaged(&random_walk_integrator, sizeof(RandomWalkIntegrator)));
        gpu_dynamic_pointers.push_back(random_walk_integrator);

        random_walk_integrator->init(integrator_base, 5);
        renderer->integrator->init(random_walk_integrator);
        return;
    }

    if (integrator_name == "simplepath") {
        SimplePathIntegrator *simple_path_integrator;
        CHECK_CUDA_ERROR(cudaMallocManaged(&simple_path_integrator, sizeof(SimplePathIntegrator)));
        gpu_dynamic_pointers.push_back(simple_path_integrator);

        simple_path_integrator->init(integrator_base, 5);
        renderer->integrator->init(simple_path_integrator);

        return;
    }

    printf("\n%s(): unknown Integrator: %s\n\n", __func__, integrator_name.value().c_str());
    REPORT_FATAL_ERROR();
}

void SceneBuilder::parse_light_source(const std::vector<Token> &tokens) {
    const auto parameters = build_parameter_dictionary(sub_vector(tokens, 2));

    auto texture_file = root + "/" + parameters.get_string("filename", {});

    auto scale = parameters.get_float("scale", 1.0);

    const Spectrum *cie_xyz[3];
    renderer->global_variables->get_cie_xyz(cie_xyz);
    const auto cie_y = cie_xyz[1];

    scale /= renderer->global_variables->rgb_color_space->illuminant->to_photometric(cie_y);

    auto image_infinite_light = ImageInfiniteLight::create(
        get_render_from_object(), texture_file, scale, renderer->global_variables->rgb_color_space,
        gpu_dynamic_pointers);

    Light *infinite_light;
    CHECK_CUDA_ERROR(cudaMallocManaged(&infinite_light, sizeof(Light)));
    gpu_dynamic_pointers.push_back(infinite_light);

    infinite_light->init(image_infinite_light);
    gpu_lights.push_back(infinite_light);
}

void SceneBuilder::parse_make_named_material(const std::vector<Token> &tokens) {
    if (tokens[0] != Token(TokenType::Keyword, "MakeNamedMaterial")) {
        REPORT_FATAL_ERROR();
    }

    const auto material_name = tokens[1].values[0];

    const auto parameters = build_parameter_dictionary(sub_vector(tokens, 2));

    auto type_of_material = parameters.get_string("type", std::nullopt);

    named_material[material_name] = create_material(type_of_material, parameters);
}

void SceneBuilder::parse_material(const std::vector<Token> &tokens) {
    if (tokens[0] != Token(TokenType::Keyword, "Material")) {
        REPORT_FATAL_ERROR();
    }

    auto type_of_material = tokens[1].values[0];

    const auto parameters = build_parameter_dictionary(sub_vector(tokens, 2));

    graphics_state.material = create_material(type_of_material, parameters);
}

void SceneBuilder::parse_named_material(const std::vector<Token> &tokens) {
    if (tokens[0] != Token(TokenType::Keyword, "NamedMaterial")) {
        REPORT_FATAL_ERROR();
    }

    auto material_name = tokens[1].values[0];

    graphics_state.material = named_material.at(material_name);
}

void SceneBuilder::parse_rotate(const std::vector<Token> &tokens) {
    if (tokens[0] != Token(TokenType::Keyword, "Rotate")) {
        REPORT_FATAL_ERROR();
    }

    std::vector<FloatType> data;
    for (int idx = 1; idx < tokens.size(); idx++) {
        data.push_back(tokens[idx].to_float());
    }

    graphics_state.transform =
        graphics_state.transform * Transform::rotate(data[0], data[1], data[2], data[3]);
}

void SceneBuilder::parse_scale(const std::vector<Token> &tokens) {
    if (tokens[0] != Token(TokenType::Keyword, "Scale")) {
        REPORT_FATAL_ERROR();
    }

    std::vector<FloatType> data;
    for (int idx = 1; idx < tokens.size(); idx++) {
        data.push_back(tokens[idx].to_float());
    }

    graphics_state.transform *= Transform::scale(data[0], data[1], data[2]);
}

void SceneBuilder::parse_area_light_source(const std::vector<Token> &tokens) {
    if (tokens[0] != Token(TokenType::Keyword, "AreaLightSource")) {
        REPORT_FATAL_ERROR();
    }

    if (tokens[1] != Token(TokenType::String, "diffuse")) {
        throw std::runtime_error("parse_area_light_source: only `diffuse` supported at the moment");
    }

    graphics_state.area_light_entity = AreaLightEntity(
        tokens[1].values[0],
        ParameterDict(sub_vector(tokens, 2), pre_computed_spectrum.named_spectra,
                      named_spectrum_texture, root, renderer->global_variables->rgb_color_space));
}

void SceneBuilder::parse_shape(const std::vector<Token> &tokens) {
    if (tokens[0] != Token(TokenType::Keyword, "Shape")) {
        REPORT_FATAL_ERROR();
    }

    const auto parameters = build_parameter_dictionary(sub_vector(tokens, 2));

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

    if (type_of_shape == "sphere") {
        auto sphere = Shape::create_sphere(
            get_render_from_object(), get_render_from_object().inverse(),
            graphics_state.reverse_orientation, parameters, gpu_dynamic_pointers);

        if (graphics_state.area_light_entity) {
            auto diffuse_light = Light::create_diffuse_area_light(
                get_render_from_object(), graphics_state.area_light_entity->parameters, sphere,
                renderer->global_variables, gpu_dynamic_pointers);

            gpu_lights.push_back(diffuse_light);

            GeometricPrimitive *geometric_primitive;
            CHECK_CUDA_ERROR(cudaMallocManaged(&geometric_primitive, sizeof(GeometricPrimitive)));
            Primitive *primitive;
            CHECK_CUDA_ERROR(cudaMallocManaged(&primitive, sizeof(Primitive)));

            gpu_dynamic_pointers.push_back(geometric_primitive);
            gpu_dynamic_pointers.push_back(primitive);

            geometric_primitive->init(sphere, graphics_state.material, diffuse_light);
            primitive->init(geometric_primitive);

            gpu_primitives.push_back(primitive);
        } else {
            auto primitive = Primitive::create_simple_primitive(sphere, graphics_state.material,
                                                                gpu_dynamic_pointers);
            gpu_primitives.push_back(primitive);
        }

        return;
    }

    if (type_of_shape == "disk") {
        printf("\nignore Shape::%s for the moment\n\n", type_of_shape.c_str());
        return;
    }

    std::cout << "\n" << __func__ << "(): unknown shape: `" << type_of_shape << "`\n\n\n";
    REPORT_FATAL_ERROR();
}

void SceneBuilder::parse_texture(const std::vector<Token> &tokens) {
    auto texture_name = tokens[1].values[0];
    auto color_type = tokens[2].values[0];
    auto texture_type = tokens[3].values[0];

    if (color_type == "spectrum") {
        const auto parameters = build_parameter_dictionary(sub_vector(tokens, 4));

        if (texture_type == "imagemap") {
            auto image_texture = SpectrumImageTexture::create(
                parameters, gpu_dynamic_pointers, renderer->global_variables->rgb_color_space);

            SpectrumTexture *spectrum_texture;
            CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_texture, sizeof(SpectrumTexture)));
            spectrum_texture->init(image_texture);

            named_spectrum_texture[texture_name] = spectrum_texture;

            gpu_dynamic_pointers.push_back(spectrum_texture);
            return;
        }

        if (texture_type == "scale") {
            SpectrumScaleTexture *scale_texture;
            SpectrumTexture *spectrum_texture;

            CHECK_CUDA_ERROR(cudaMallocManaged(&scale_texture, sizeof(SpectrumScaleTexture)));
            CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_texture, sizeof(SpectrumTexture)));

            scale_texture->init(parameters);
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

void SceneBuilder::parse_transform(const std::vector<Token> &tokens) {
    if (tokens[0] != Token(TokenType::Keyword, "Transform")) {
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

    graphics_state.transform = transform_matrix.transpose();
}

void SceneBuilder::parse_translate(const std::vector<Token> &tokens) {
    std::vector<FloatType> data;
    for (int idx = 1; idx < tokens.size(); idx++) {
        data.push_back(tokens[idx].to_float());
    }

    assert(data.size() == 16);

    graphics_state.transform *= Transform::translate(data[0], data[1], data[2]);
}

void SceneBuilder::add_triangle_mesh(const std::vector<Point3f> &points,
                                     const std::vector<int> &indices,
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
                simple_primitives, shapes, graphics_state.material, num_triangles);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            GPU::init_primitives<<<blocks, threads>>>(primitives, simple_primitives, num_triangles);
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
            CHECK_CUDA_ERROR(
                cudaMallocManaged(&diffuse_area_lights, sizeof(DiffuseAreaLight) * num_triangles));

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

            GPU::init_geometric_primitives<<<blocks, threads>>>(
                geometric_primitives, shapes, graphics_state.material, lights, num_triangles);
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

void SceneBuilder::parse_tokens(const std::vector<Token> &tokens) {
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

        graphics_state.transform = Transform::identity();
        named_coordinate_systems["world"] = graphics_state.transform;

        return;
    }

    case TokenType::Keyword: {
        const auto keyword = first_token.values[0];

        if (integrator_name == "ambientocclusion" || integrator_name == "surfacenormal") {
            static std::set<std::string> ignored_keywords;
            if (keyword == "AreaLightSource" || keyword == "LightSource" || keyword == "Material" ||
                keyword == "Texture") {

                if (ignored_keywords.find(keyword) == ignored_keywords.end()) {
                    ignored_keywords.insert(keyword);
                    printf("Integrator `%s`: ignore keyword `%s`\n",
                           integrator_name.value().c_str(), keyword.c_str());
                }

                return;
            }
        }

        if (keyword == "AreaLightSource") {
            parse_area_light_source(tokens);
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

        if (keyword == "LightSource") {
            parse_light_source(tokens);
            return;
        }

        if (keyword == "LookAt") {
            parse_lookat(tokens);
            return;
        }

        if (keyword == "MakeNamedMaterial") {
            parse_make_named_material(tokens);
            return;
        }

        if (keyword == "Material") {
            parse_material(tokens);
            return;
        }

        if (keyword == "NamedMaterial") {
            parse_named_material(tokens);
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

        if (keyword == "ConcatTransform" || keyword == "MakeNamedMedium" ||
            keyword == "MediumInterface" || keyword == "ObjectBegin" || keyword == "ObjectEnd" ||
            keyword == "ObjectInstance" || keyword == "PixelFilter") {

            static std::set<std::string> unimplemented_keywords;
            if (unimplemented_keywords.find(keyword) == unimplemented_keywords.end()) {
                unimplemented_keywords.insert(keyword);
                printf("%s(): keyword `%s` not implemented\n", __func__, keyword.c_str());
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

void SceneBuilder::parse_file(const std::string &_filename) {
    const auto all_tokens = parse_pbrt_into_token(_filename);
    const auto range_of_tokens = group_tokens(all_tokens);

    for (uint range_idx = 0; range_idx < range_of_tokens.size() - 1; ++range_idx) {
        auto current_tokens = std::vector(all_tokens.begin() + range_of_tokens[range_idx],
                                          all_tokens.begin() + range_of_tokens[range_idx + 1]);

        parse_tokens(current_tokens);
    }
}

void SceneBuilder::preprocess() {
    renderer->bvh->build_bvh(gpu_primitives, gpu_dynamic_pointers, thread_pool);

    auto full_bounds = renderer->bvh->bounds();
    for (auto light : gpu_lights) {
        light->preprocess(full_bounds);
    }

    build_integrator();
}

void SceneBuilder::render() const {
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
