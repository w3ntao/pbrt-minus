#include "pbrt/scene/scene_builder.h"

#include <set>

#include "pbrt/accelerator/hlbvh.h"

#include "pbrt/base/film.h"
#include "pbrt/base/filter.h"
#include "pbrt/base/integrator.h"
#include "pbrt/base/material.h"
#include "pbrt/base/sampler.h"
#include "pbrt/base/shape.h"
#include "pbrt/base/primitive.h"

#include "pbrt/gpu/renderer.h"

#include "pbrt/integrators/integrator_base.h"

#include "pbrt/light_samplers/uniform_light_sampler.h"
#include "pbrt/light_samplers/power_light_sampler.h"

#include "pbrt/spectrum_util/global_spectra.h"
#include "pbrt/spectrum_util/rgb_color_space.h"
#include "pbrt/spectrum_util/spectrum_constants_metal.h"

#include "pbrt/textures/spectrum_constant_texture.h"

#include "pbrt/util/std_container.h"

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
      output_filename(command_line_option.output_file) {

    global_spectra =
        GlobalSpectra::create(RGBtoSpectrumData::Gamut::sRGB, thread_pool, gpu_dynamic_pointers);

    auto ag_eta = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(Ag_eta), std::end(Ag_eta)), false, nullptr, gpu_dynamic_pointers);

    auto ag_k = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(Ag_k), std::end(Ag_k)), false, nullptr, gpu_dynamic_pointers);

    named_spectra = {
        {"metal-Ag-eta", ag_eta},
        {"metal-Ag-k", ag_k},
    };

    renderer = Renderer::create(gpu_dynamic_pointers);

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

        renderer->camera = Camera::create_perspective_camera(
            film_resolution.value(), camera_transform, fov, 0.0, gpu_dynamic_pointers);

        return;
    }

    printf("\n%s(): Camera type `%s` not implemented.\n", __func__, camera_type.c_str());
    REPORT_FATAL_ERROR();
}

void SceneBuilder::build_filter() {
    renderer->filter = Filter::create_box_filter(0.5, gpu_dynamic_pointers);
}

void SceneBuilder::build_film() {
    const auto parameters = build_parameter_dictionary(sub_vector(film_tokens, 2));

    auto resolution_x = parameters.get_integer("xresolution")[0];
    auto resolution_y = parameters.get_integer("yresolution")[0];

    film_resolution = Point2i(resolution_x, resolution_y);

    if (output_filename.empty()) {
        output_filename = parameters.get_string("filename", std::nullopt);
    }

    if (std::filesystem::path p(output_filename); p.extension() != ".png") {
        printf("output filename extension: only PNG is supported for the moment\n");
        output_filename = p.replace_extension(".png").filename();
    }

    renderer->film = Film::create_rgb_film(parameters, gpu_dynamic_pointers);
}

void SceneBuilder::build_sampler() {
    // TODO: sampler is not parsed, only pixelsamples read
    const auto parameters = build_parameter_dictionary(sub_vector(sampler_tokens, 2));

    auto samples_from_parameters = parameters.get_integer("pixelsamples");

    if (!samples_per_pixel.has_value()) {
        if (!samples_from_parameters.empty()) {
            samples_per_pixel = samples_from_parameters[0];
        } else {
            samples_per_pixel = 4;
            // default samples per pixel
        }
    }

    const std::string type_sampler = "stratified";
    // const std::string type_sampler = "independent";
    uint total_pixel_num = film_resolution->x * film_resolution->y;

    renderer->samplers = Sampler::create(type_sampler, samples_per_pixel.value(), total_pixel_num,
                                         gpu_dynamic_pointers);
    samples_per_pixel = renderer->samplers->get_samples_per_pixel();
}

void SceneBuilder::build_integrator() {
    const auto parameters = build_parameter_dictionary(sub_vector(integrator_tokens, 2));

    IntegratorBase *integrator_base;
    CHECK_CUDA_ERROR(cudaMallocManaged(&integrator_base, sizeof(IntegratorBase)));

    const Light **light_array;
    CHECK_CUDA_ERROR(cudaMallocManaged(&light_array, sizeof(Light *) * gpu_lights.size()));
    CHECK_CUDA_ERROR(cudaMemcpy(light_array, gpu_lights.data(), sizeof(Light *) * gpu_lights.size(),
                                cudaMemcpyHostToDevice));
    
    auto power_light_sampler =
        PowerLightSampler::create(light_array, gpu_lights.size(), gpu_dynamic_pointers);

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

    integrator_base->light_sampler = power_light_sampler;

    integrator_base->infinite_lights = gpu_infinite_lights;
    integrator_base->infinite_light_num = infinite_lights.size();

    for (auto ptr : std::vector<void *>({
             integrator_base,
             light_array,
             gpu_infinite_lights,
         })) {
        gpu_dynamic_pointers.push_back(ptr);
    }

    renderer->integrator =
        Integrator::create(parameters, integrator_name, integrator_base, gpu_dynamic_pointers);
}

void SceneBuilder::parse_light_source(const std::vector<Token> &tokens) {
    const auto parameters = build_parameter_dictionary(sub_vector(tokens, 2));

    const auto light_source_type = tokens[1].values[0];

    auto light = Light::create(light_source_type, get_render_from_object(), parameters,
                               gpu_dynamic_pointers);
    gpu_lights.push_back(light);
}

void SceneBuilder::parse_make_named_material(const std::vector<Token> &tokens) {
    if (tokens[0] != Token(TokenType::Keyword, "MakeNamedMaterial")) {
        REPORT_FATAL_ERROR();
    }

    const auto material_name = tokens[1].values[0];

    const auto parameters = build_parameter_dictionary(sub_vector(tokens, 2));

    auto type_of_material = parameters.get_string("type", std::nullopt);

    named_material[material_name] =
        Material::create(type_of_material, parameters, gpu_dynamic_pointers);
}

void SceneBuilder::parse_material(const std::vector<Token> &tokens) {
    if (tokens[0] != Token(TokenType::Keyword, "Material")) {
        REPORT_FATAL_ERROR();
    }

    auto type_of_material = tokens[1].values[0];

    const auto parameters = build_parameter_dictionary(sub_vector(tokens, 2));

    graphics_state.material = Material::create(type_of_material, parameters, gpu_dynamic_pointers);
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

    graphics_state.area_light_entity =
        AreaLightEntity(tokens[1].values[0], build_parameter_dictionary(sub_vector(tokens, 2)));
}

void SceneBuilder::parse_shape(const std::vector<Token> &tokens) {
    if (tokens[0] != Token(TokenType::Keyword, "Shape")) {
        REPORT_FATAL_ERROR();
    }

    const auto parameters = build_parameter_dictionary(sub_vector(tokens, 2));
    auto type_of_shape = tokens[1].values[0];
    const auto render_from_object = get_render_from_object();

    auto result =
        Shape::create(type_of_shape, render_from_object, render_from_object.inverse(),
                      graphics_state.reverse_orientation, parameters, gpu_dynamic_pointers);
    auto shapes = result.first;
    auto num_shapes = result.second;

    if (!graphics_state.area_light_entity) {
        auto simple_primitives = Primitive::create_simple_primitives(
            shapes, graphics_state.material, num_shapes, gpu_dynamic_pointers);

        for (uint idx = 0; idx < num_shapes; ++idx) {
            gpu_primitives.push_back(&simple_primitives[idx]);
        }

        return;
    }

    auto diffuse_area_lights = Light::create_diffuse_area_lights(
        shapes, num_shapes, render_from_object, graphics_state.area_light_entity->parameters,
        gpu_dynamic_pointers);

    auto geometric_primitives = Primitive::create_geometric_primitives(
        shapes, graphics_state.material, diffuse_area_lights, num_shapes, gpu_dynamic_pointers);

    // otherwise: build AreaDiffuseLight
    for (uint idx = 0; idx < num_shapes; ++idx) {
        gpu_lights.push_back(&diffuse_area_lights[idx]);
        gpu_primitives.push_back(&geometric_primitives[idx]);
    }
}

void SceneBuilder::parse_texture(const std::vector<Token> &tokens) {
    auto texture_name = tokens[1].values[0];
    auto color_type = tokens[2].values[0];
    auto texture_type = tokens[3].values[0];
    const auto parameters = build_parameter_dictionary(sub_vector(tokens, 4));

    if (color_type == "spectrum") {
        auto spectrum_texture = SpectrumTexture::create(
            texture_type, parameters, global_spectra->rgb_color_space, gpu_dynamic_pointers);

        named_spectrum_texture[texture_name] = spectrum_texture;

        return;
    }

    printf("\n%s(): color type `%s` not implemented\n", __func__, color_type.c_str());
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

    graphics_state.transform *= Transform::translate(data[0], data[1], data[2]);
}

void SceneBuilder::parse_tokens(const std::vector<Token> &tokens) {
    if (tokens.empty()) {
        REPORT_FATAL_ERROR();
    }

    const Token &first_token = tokens[0];

    switch (first_token.type) {
    case TokenType::AttributeBegin: {
        pushed_graphics_state.push(graphics_state);
        return;
    }

    case TokenType::AttributeEnd: {
        if (pushed_graphics_state.empty()) {
            REPORT_FATAL_ERROR();
        }

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

        if (this->should_ignore_material_and_texture()) {
            static std::set<std::string> ignored_keywords;
            if (keyword == "AreaLightSource" || keyword == "LightSource" || keyword == "Material" ||
                keyword == "MakeNamedMaterial" || keyword == "NamedMaterial" ||
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

            auto camera_from_world = graphics_state.transform;
            named_coordinate_systems["camera"] = camera_from_world.inverse();

            return;
        }

        if (keyword == "CoordSysTransform") {
            auto coord_sys_name = tokens[1].values[0];
            if (named_coordinate_systems.find(coord_sys_name) == named_coordinate_systems.end()) {
                printf("\ncoordinate system `%s` not available\n", coord_sys_name.c_str());
                REPORT_FATAL_ERROR();
            }

            graphics_state.transform = named_coordinate_systems.at(coord_sys_name);
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
            integrator_tokens = tokens;
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
    renderer->bvh = HLBVH::create(gpu_primitives, gpu_dynamic_pointers, thread_pool);

    auto full_bounds = renderer->bvh->bounds();
    for (auto light : gpu_lights) {
        light->preprocess(full_bounds);
    }

    build_integrator();
}

void SceneBuilder::render() const {
    auto start = std::chrono::system_clock::now();

    renderer->render(output_filename, film_resolution.value(), samples_per_pixel.value());

    const std::chrono::duration<FloatType> duration{std::chrono::system_clock::now() - start};

    std::cout << std::fixed << std::setprecision(1) << "rendering took " << duration.count()
              << " seconds.\n"
              << std::flush;

    std::cout << "image saved to `" << output_filename << "`\n";
}
