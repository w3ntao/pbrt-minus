#include "pbrt/scene/scene_builder.h"

#include <set>

#include "pbrt/accelerator/hlbvh.h"

#include "pbrt/base/float_texture.h"
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
#include "pbrt/spectrum_util/spectrum_constants_glass.h"

#include "pbrt/textures/spectrum_constant_texture.h"

#include "pbrt/util/std_container.h"

#include "pbrt/wavefront/path.h"

uint next_keyword_position(const std::vector<Token> &tokens, uint start) {
    for (uint idx = start + 1; idx < tokens.size(); ++idx) {
        const auto type = tokens[idx].type;
        if (type == TokenType::Number || type == TokenType::String || type == TokenType::Variable ||
            type == TokenType::List) {
            continue;
        }
        return idx;
    }

    return tokens.size();
}

void add_one_to_map(const std::string &key, std::map<std::string, uint> &counter) {
    if (counter.find(key) == counter.end()) {
        counter[key] = 1;
        return;
    }

    counter[key] += 1;
}

std::map<std::string, uint> count_light_type(const std::vector<Light *> &gpu_lights) {
    std::map<std::string, uint> counter;
    for (const auto light : gpu_lights) {
        switch (light->type) {
        case (Light::Type::diffuse_area_light): {
            add_one_to_map("DiffuseAreaLight", counter);
            break;
        }

        case (Light::Type::distant_light): {
            add_one_to_map("DistantLight", counter);
            break;
        }

        case (Light::Type::image_infinite_light): {
            add_one_to_map("ImageInfiniteLight", counter);
            break;
        }
        default: {
            REPORT_FATAL_ERROR();
        }
        }
    }

    return counter;
}

std::map<std::string, uint> count_material_type(const std::vector<const Primitive *> &primitives) {
    std::map<std::string, uint> counter;
    for (const auto primitive : primitives) {
        auto material_type = primitive->get_material()->get_material_type();

        switch (material_type) {
        case (Material::Type::coated_conductor): {
            add_one_to_map("CoatedConductor", counter);
            break;
        }

        case (Material::Type::coated_diffuse): {
            add_one_to_map("CoatedDiffuse", counter);
            break;
        }

        case (Material::Type::conductor): {
            add_one_to_map("Conductor", counter);
            break;
        }

        case (Material::Type::diffuse): {
            add_one_to_map("Diffuse", counter);
            break;
        }

        case (Material::Type::dielectric): {
            add_one_to_map("Dielectric", counter);
            break;
        }

        case (Material::Type::mix): {
            // TODO: count different mixed material
            add_one_to_map("Mix", counter);
            break;
        }

        default: {
            REPORT_FATAL_ERROR();
        }
        }
    }

    return counter;
}

SceneBuilder::SceneBuilder(const CommandLineOption &command_line_option)
    : integrator_name(command_line_option.integrator_name),
      output_filename(command_line_option.output_file),
      samples_per_pixel(command_line_option.samples_per_pixel) {

    global_spectra =
        GlobalSpectra::create(RGBtoSpectrumData::Gamut::sRGB, thread_pool, gpu_dynamic_pointers);

    auto ag_eta = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(Ag_eta), std::end(Ag_eta)), false, nullptr, gpu_dynamic_pointers);

    auto ag_k = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(Ag_k), std::end(Ag_k)), false, nullptr, gpu_dynamic_pointers);

    auto al_eta = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(Al_eta), std::end(Al_eta)), false, nullptr, gpu_dynamic_pointers);

    auto al_k = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(Al_k), std::end(Al_k)), false, nullptr, gpu_dynamic_pointers);

    auto au_eta = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(Au_eta), std::end(Au_eta)), false, nullptr, gpu_dynamic_pointers);

    auto au_k = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(Au_k), std::end(Au_k)), false, nullptr, gpu_dynamic_pointers);

    auto cu_eta = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(Cu_eta), std::end(Cu_eta)), false, nullptr, gpu_dynamic_pointers);

    auto cu_k = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(Cu_k), std::end(Cu_k)), false, nullptr, gpu_dynamic_pointers);

    auto glass_bk7_eta = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(GlassBK7_eta), std::end(GlassBK7_eta)), false, nullptr,
        gpu_dynamic_pointers);

    spectra = {
        {"metal-Ag-eta", ag_eta}, {"metal-Ag-k", ag_k},     {"metal-Al-eta", al_eta},
        {"metal-Al-k", al_k},     {"metal-Au-eta", au_eta}, {"metal-Au-k", au_k},
        {"metal-Cu-eta", cu_eta}, {"metal-Cu-k", cu_k},     {"glass-BK7", glass_bk7_eta},
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

        renderer->camera = Camera::create_perspective_camera(
            film_resolution.value(), camera_transform, parameters, gpu_dynamic_pointers);

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

    auto resolution_x = parameters.get_integer("xresolution");
    auto resolution_y = parameters.get_integer("yresolution");

    film_resolution = Point2i(resolution_x, resolution_y);

    if (output_filename.empty()) {
        output_filename = parameters.get_one_string("filename");
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

    if (!samples_per_pixel.has_value()) {
        samples_per_pixel = parameters.get_integer("pixelsamples", 4);
    }
    // TODO: implement stratified sampler for wavefront renderer
    //  const std::string type_sampler = "stratified";
    const std::string type_sampler = "independent";
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

    auto light_sampler =
        PowerLightSampler::create(light_array, gpu_lights.size(), gpu_dynamic_pointers);

    /*
    auto light_sampler =
        UniformLightSampler::create(light_array, gpu_lights.size(), gpu_dynamic_pointers);
    */

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

    integrator_base->light_sampler = light_sampler;

    integrator_base->infinite_lights = gpu_infinite_lights;
    integrator_base->infinite_light_num = infinite_lights.size();

    for (auto ptr : std::vector<void *>({
             integrator_base,
             light_array,
             gpu_infinite_lights,
         })) {
        gpu_dynamic_pointers.push_back(ptr);
    }

    renderer->wavefront_path_integrator = WavefrontPathIntegrator::create(
        parameters, integrator_base, samples_per_pixel.value(), gpu_dynamic_pointers);

    return;

    // TODO: automatically choose wavefront rendering or mega-kernel rendering

    renderer->integrator =
        Integrator::create(parameters, integrator_name, integrator_base, gpu_dynamic_pointers);
}

void SceneBuilder::parse_keyword(const std::vector<Token> &tokens) {
    const auto keyword = tokens[0].values[0];

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

    if (keyword == "ConcatTransform") {
        parse_concat_transform(tokens);
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

    if (keyword == "Identity") {
        graphics_state.transform = Transform::identity();
        return;
    }

    if (keyword == "Include") {
        auto included_file = tokens[1].values[0];
        parse_file(get_file_full_path(included_file));
        return;
    }

    if (keyword == "Integrator") {
        if (integrator_name.has_value()) {
            // ignore config file, when integrator is read from command line option
            return;
        }

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

    if (keyword == "MakeNamedMedium" || keyword == "MediumInterface" || keyword == "PixelFilter") {

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

void SceneBuilder::parse_concat_transform(const std::vector<Token> &tokens) {
    if (tokens[0] != Token(TokenType::Keyword, "ConcatTransform")) {
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

    graphics_state.transform = graphics_state.transform * transform_matrix.transpose();
}

void SceneBuilder::parse_light_source(const std::vector<Token> &tokens) {
    const auto parameters = build_parameter_dictionary(sub_vector(tokens, 2));

    const auto light_source_type = tokens[1].values[0];

    auto light = Light::create(light_source_type, get_render_from_object(), parameters,
                               gpu_dynamic_pointers);
    gpu_lights.push_back(light);
}

void SceneBuilder::parse_lookat(const std::vector<Token> &tokens) {
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

void SceneBuilder::parse_make_named_material(const std::vector<Token> &tokens) {
    if (tokens[0] != Token(TokenType::Keyword, "MakeNamedMaterial")) {
        REPORT_FATAL_ERROR();
    }

    const auto material_name = tokens[1].values[0];

    const auto parameters = build_parameter_dictionary(sub_vector(tokens, 2));

    auto type_of_material = parameters.get_one_string("type");

    materials[material_name] = Material::create(type_of_material, parameters, gpu_dynamic_pointers);
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

    graphics_state.material = materials.at(material_name);
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

        if (active_instance_definition) {
            active_instance_definition->instantiated_primitives.push_back(
                InstantiatedPrimitive(simple_primitives, num_shapes));
        } else {
            for (uint idx = 0; idx < num_shapes; ++idx) {
                gpu_primitives.push_back(&simple_primitives[idx]);
            }
        }

        return;
    }

    if (active_instance_definition) {
        printf("\nERROR: area lights not supported with object instancing\n");
        REPORT_FATAL_ERROR();
    }

    auto diffuse_area_lights = Light::create_diffuse_area_lights(
        shapes, num_shapes, render_from_object, graphics_state.area_light_entity->parameters,
        gpu_dynamic_pointers);

    auto geometric_primitives = Primitive::create_geometric_primitives(
        shapes, graphics_state.material, diffuse_area_lights, num_shapes, gpu_dynamic_pointers);

    // otherwise: build AreaDiffuseLight
    for (uint idx = 0; idx < num_shapes; ++idx) {
        auto primitive_ptr = &geometric_primitives[idx];
        auto area_light_ptr = &diffuse_area_lights[idx];

        gpu_lights.push_back(area_light_ptr);
        gpu_primitives.push_back(primitive_ptr);
    }
}

void SceneBuilder::parse_texture(const std::vector<Token> &tokens) {
    auto texture_name = tokens[1].values[0];
    auto color_type = tokens[2].values[0];
    auto texture_type = tokens[3].values[0];
    const auto parameters = build_parameter_dictionary(sub_vector(tokens, 4));

    if (color_type == "float") {
        auto float_texture = FloatTexture::create(texture_type, get_render_from_object(),
                                                  parameters, gpu_dynamic_pointers);
        float_textures[texture_name] = float_texture;

        return;
    }

    if (color_type == "spectrum") {
        albedo_spectrum_textures[texture_name] = SpectrumTexture::create(
            texture_type, SpectrumType::Albedo, get_render_from_object(),
            global_spectra->rgb_color_space, parameters, gpu_dynamic_pointers);

        illuminant_spectrum_textures[texture_name] = SpectrumTexture::create(
            texture_type, SpectrumType::Illuminant, get_render_from_object(),
            global_spectra->rgb_color_space, parameters, gpu_dynamic_pointers);

        unbounded_spectrum_textures[texture_name] = SpectrumTexture::create(
            texture_type, SpectrumType::Unbounded, get_render_from_object(),
            global_spectra->rgb_color_space, parameters, gpu_dynamic_pointers);

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
    uint token_idx = 0;
    while (token_idx < tokens.size()) {
        const Token &first_token = tokens[token_idx];
        if (first_token.type == TokenType::WorldBegin) {
            build_film();
            build_filter();
            build_camera();
            build_sampler();

            graphics_state.transform = Transform::identity();
            named_coordinate_systems["world"] = graphics_state.transform;

            token_idx += 1;
            continue;
        }

        if (first_token.type == TokenType::AttributeBegin) {
            pushed_graphics_state.push(graphics_state);

            token_idx += 1;
            continue;
        }

        if (first_token.type == TokenType::AttributeEnd) {
            if (pushed_graphics_state.empty()) {
                REPORT_FATAL_ERROR();
            }

            graphics_state = pushed_graphics_state.top();
            pushed_graphics_state.pop();

            token_idx += 1;
            continue;
        }

        if (first_token.type == TokenType::ObjectBegin) {
            pushed_graphics_state.push(graphics_state);

            if (active_instance_definition) {
                printf("\nERROR: ObjectBegin called inside of instance definition\n");
                REPORT_FATAL_ERROR();
            }

            active_instance_definition = std::make_shared<ActiveInstanceDefinition>();

            active_instance_definition->name = first_token.values[0];

            token_idx += 1;
            continue;
        }

        if (first_token.type == TokenType::ObjectEnd) {
            if (!active_instance_definition) {
                printf("\nERROR: ObjectEnd called before an instance defined\n");
                REPORT_FATAL_ERROR();
            }

            instance_definition[active_instance_definition->name] = active_instance_definition;

            active_instance_definition = nullptr;

            graphics_state = pushed_graphics_state.top();
            pushed_graphics_state.pop();

            token_idx += 1;
            continue;
        }

        if (first_token.type == TokenType::ObjectInstance) {
            const auto object_name = first_token.values[0];
            if (instance_definition.find(object_name) == instance_definition.end()) {
                printf("\nERROR: object `%s` not found\n", object_name.c_str());
                REPORT_FATAL_ERROR();
            }

            const auto instance = instance_definition.at(object_name);

            auto world_from_render = render_from_world.inverse();
            auto render_from_instance = get_render_from_object() * world_from_render;

            if (render_from_instance.is_identity()) {
                for (auto &instanced_primitives : instance->instantiated_primitives) {
                    for (uint p_idx = 0; p_idx < instanced_primitives.num; ++p_idx) {
                        gpu_primitives.push_back(&instanced_primitives.primitives[p_idx]);
                    }
                }
            } else {
                for (auto &instanced_primitives : instance->instantiated_primitives) {
                    auto transformed_primitives = Primitive::create_transformed_primitives(
                        instanced_primitives.primitives, render_from_instance,
                        instanced_primitives.num, gpu_dynamic_pointers);

                    for (uint p_idx = 0; p_idx < instanced_primitives.num; ++p_idx) {
                        gpu_primitives.push_back(&transformed_primitives[p_idx]);
                    }
                }
            }

            token_idx += 1;
            continue;
        }

        if (first_token.type == TokenType::Keyword) {
            auto end = next_keyword_position(tokens, token_idx);
            auto keyword_tokens = sub_vector(tokens, token_idx, end);

            parse_keyword(keyword_tokens);

            token_idx = end;
            continue;
        }

        std::cout << "\nillegal token: \n" << first_token << "\n";
        REPORT_FATAL_ERROR();
    }
}

void SceneBuilder::parse_file(const std::string &_filename) {
    const auto all_tokens = parse_pbrt_into_token(_filename);
    parse_tokens(all_tokens);
}

void SceneBuilder::preprocess() {
    renderer->bvh = HLBVH::create(gpu_primitives, gpu_dynamic_pointers, thread_pool);

    auto full_scene_bounds = renderer->bvh->bounds();
    for (auto light : gpu_lights) {
        light->preprocess(full_scene_bounds);
    }
    build_integrator();

    printf("\n");
    if (renderer->wavefront_path_integrator != nullptr) {
        printf("Integrator: wavefront path integrator\n");

    } else if (renderer->integrator != nullptr) {
        printf("Integrator: %s\n", renderer->integrator->get_name().c_str());
    } else {
        REPORT_FATAL_ERROR();
    }

    printf("Sampler: %s\n", renderer->samplers->get_name().c_str());
    printf("\n");

    auto light_type_counter = count_light_type(gpu_lights);

    auto light_size = gpu_lights.size();
    printf("total lights: %zu\n", light_size);
    for (auto const &kv : light_type_counter) {
        printf("    %s: %d (%.2f%)\n", kv.first.c_str(), kv.second,
               double(kv.second) / light_size * 100);
    }
    printf("\n");

    auto primitives_size = gpu_primitives.size();
    auto material_type_counter = count_material_type(gpu_primitives);
    printf("materials' type: %zu\n", material_type_counter.size());
    for (auto const &kv : material_type_counter) {
        printf("    %s: %d (%.2f%)\n", kv.first.c_str(), kv.second,
               double(kv.second) / primitives_size * 100);
    }
    printf("\n");
}

void SceneBuilder::render() const {
    auto start = std::chrono::system_clock::now();

    renderer->render(output_filename, samples_per_pixel.value());

    const std::chrono::duration<FloatType> duration{std::chrono::system_clock::now() - start};

    std::cout << std::fixed << std::setprecision(1) << "rendering took " << duration.count()
              << " seconds.\n"
              << std::flush;

    std::cout << "image saved to `" << output_filename << "`\n";
}
