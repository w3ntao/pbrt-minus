#include <pbrt/accelerator/hlbvh.h>
#include <pbrt/base/film.h>
#include <pbrt/base/filter.h>
#include <pbrt/base/float_texture.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/material.h>
#include <pbrt/base/megakernel_integrator.h>
#include <pbrt/base/primitive.h>
#include <pbrt/base/sampler.h>
#include <pbrt/base/shape.h>
#include <pbrt/films/grey_scale_film.h>
#include <pbrt/integrators/bdpt.h>
#include <pbrt/integrators/mlt_bdpt.h>
#include <pbrt/integrators/mlt_path.h>
#include <pbrt/integrators/wavefront_path.h>
#include <pbrt/light_samplers/power_light_sampler.h>
#include <pbrt/light_samplers/uniform_light_sampler.h>
#include <pbrt/primitives/transformed_primitive.h>
#include <pbrt/scene/scene_builder.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/spectrum_util/spectrum_constants_glass.h>
#include <pbrt/spectrum_util/spectrum_constants_metal.h>
#include <pbrt/util/std_container.h>
#include <set>

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

std::map<std::string, uint> count_light_type(const std::vector<Light *> &gpu_lights) {
    const std::map<Light::Type, std::string> lights_name = {
        {Light::Type::diffuse_area_light, "DiffuseAreaLight"},
        {Light::Type::distant_light, "DistantLight"},
        {Light::Type::image_infinite_light, "ImageInfiniteLight"},
        {Light::Type::spot_light, "SpotLight"},
        {Light::Type::uniform_infinite_light, "UniformInfiniteLight"},
    };

    std::map<std::string, uint> counter;

    for (const auto light : gpu_lights) {
        const auto type = light->type;
        if (lights_name.find(type) == lights_name.end()) {
            REPORT_FATAL_ERROR();
        }

        auto name = lights_name.at(type);

        if (counter.find(name) == counter.end()) {
            counter[name] = 1;
            continue;
        }

        counter[name] += 1;
    }

    return counter;
}

void SceneBuilder::ActiveInstanceDefinition::build_bvh(GPUMemoryAllocator &allocator) {
    if (bvh_build) {
        REPORT_FATAL_ERROR();
    }

    if (this->primitives.empty()) {
        return;
    }

    bvh_build = true;

    if (primitives.size() == 1) {
        return;
    }

    auto bvh = HLBVH::create(primitives, "for instance `" + this->name + "`", allocator);

    auto root = allocator.allocate<Primitive>();
    root->init(bvh);

    primitives = {root};
}

SceneBuilder::SceneBuilder(const CommandLineOption &command_line_option)
    : integrator_name(command_line_option.integrator_name),
      output_filename(command_line_option.output_file),
      samples_per_pixel(command_line_option.samples_per_pixel),
      preview(command_line_option.preview) {
    global_spectra = GlobalSpectra::create(RGBtoSpectrumData::Gamut::sRGB, allocator);

    auto ag_eta = Spectrum::create_piecewise_linear_spectrum_from_interleaved(Ag_eta, false,
                                                                              nullptr, allocator);

    auto ag_k = Spectrum::create_piecewise_linear_spectrum_from_interleaved(Ag_k, false, nullptr,
                                                                            allocator);

    auto al_eta = Spectrum::create_piecewise_linear_spectrum_from_interleaved(Al_eta, false,
                                                                              nullptr, allocator);

    auto al_k = Spectrum::create_piecewise_linear_spectrum_from_interleaved(Al_k, false, nullptr,
                                                                            allocator);

    auto au_eta = Spectrum::create_piecewise_linear_spectrum_from_interleaved(Au_eta, false,
                                                                              nullptr, allocator);

    auto au_k = Spectrum::create_piecewise_linear_spectrum_from_interleaved(Au_k, false, nullptr,
                                                                            allocator);

    auto cu_eta = Spectrum::create_piecewise_linear_spectrum_from_interleaved(Cu_eta, false,
                                                                              nullptr, allocator);

    auto cu_k = Spectrum::create_piecewise_linear_spectrum_from_interleaved(Cu_k, false, nullptr,
                                                                            allocator);

    auto glass_bk7_eta = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        GlassBK7_eta, false, nullptr, allocator);

    auto glass_f11_eta = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        GlassSF11_eta, false, nullptr, allocator);

    spectra = {
        {"metal-Ag-eta", ag_eta},     {"metal-Ag-k", ag_k},         {"metal-Al-eta", al_eta},
        {"metal-Al-k", al_k},         {"metal-Au-eta", au_eta},     {"metal-Au-k", au_k},
        {"metal-Cu-eta", cu_eta},     {"metal-Cu-k", cu_k},

        {"glass-BK7", glass_bk7_eta}, {"glass-F11", glass_f11_eta},
    };

    integrator_base = allocator.allocate<IntegratorBase>();
    integrator_base->init();

    auto texture = SpectrumTexture::create_constant_float_val_texture(0.5, allocator);
    graphics_state.material = Material::create_diffuse_material(texture, allocator);
}

void SceneBuilder::build_camera() {
    if (film == nullptr) {
        REPORT_FATAL_ERROR();
    }

    const auto parameters = build_parameter_dictionary(sub_vector(camera_tokens, 2));

    const auto camera_type = camera_tokens[1].values[0];
    if (camera_type == "perspective") {
        auto camera_from_world = graphics_state.transform;
        auto world_from_camera = camera_from_world.inverse();

        named_coordinate_systems["camera"] = world_from_camera;

        auto camera_transform =
            CameraTransform(world_from_camera, RenderingCoordinateSystem::CameraWorldCoordSystem);

        render_from_world = camera_transform.render_from_world;

        if (this->film == nullptr || integrator_base->filter == nullptr) {
            REPORT_FATAL_ERROR();
        }

        integrator_base->camera =
            Camera::create_perspective_camera(film->get_resolution(), camera_transform, this->film,
                                              integrator_base->filter, parameters, allocator);

        return;
    }

    printf("\n%s(): Camera type `%s` not implemented\n", __func__, camera_type.c_str());
    REPORT_FATAL_ERROR();
}

void SceneBuilder::build_filter() {
    ParameterDictionary parameters;
    std::string filter_type = "mitchell";
    if (!pixel_filter_tokens.empty()) {
        parameters = build_parameter_dictionary(sub_vector(pixel_filter_tokens, 2));
        filter_type = pixel_filter_tokens[1].values[0];
    }

    integrator_base->filter = Filter::create(filter_type, parameters, allocator);
}

void SceneBuilder::build_film() {
    const auto parameters = build_parameter_dictionary(sub_vector(film_tokens, 2));

    if (output_filename.empty()) {
        output_filename = parameters.get_one_string("filename");
    }

    if (std::filesystem::path p(output_filename); p.extension() != ".png") {
        printf("output filename extension: only PNG is supported for the moment\n");
        output_filename = p.replace_extension(".png").filename();
    }

    if (integrator_base->filter == nullptr) {
        REPORT_FATAL_ERROR();
    }
    film = Film::create_rgb_film(integrator_base->filter, parameters, allocator);
}

void SceneBuilder::build_gpu_lights() {
    auto light_array = allocator.allocate<const Light *>(gpu_lights.size());
    CHECK_CUDA_ERROR(cudaMemcpy(light_array, gpu_lights.data(), sizeof(Light *) * gpu_lights.size(),
                                cudaMemcpyHostToDevice));

    integrator_base->lights = light_array;
    integrator_base->light_num = gpu_lights.size();

    integrator_base->light_sampler =
        PowerLightSampler::create(light_array, gpu_lights.size(), allocator);

    std::vector<const Light *> infinite_lights;
    for (auto light : gpu_lights) {
        if (light->get_light_type() == LightType::infinite) {
            infinite_lights.push_back(light);
        }
    }

    auto gpu_infinite_lights = allocator.allocate<const Light *>(infinite_lights.size());

    CHECK_CUDA_ERROR(cudaMemcpy(gpu_infinite_lights, infinite_lights.data(),
                                sizeof(Light *) * infinite_lights.size(), cudaMemcpyHostToDevice));

    integrator_base->infinite_lights = gpu_infinite_lights;
    integrator_base->infinite_light_num = infinite_lights.size();
}

void SceneBuilder::build_integrator() {
    build_gpu_lights();

    const auto parameters = build_parameter_dictionary(sub_vector(integrator_tokens, 2));

    if (!integrator_name.has_value()) {
        integrator_name = parameters.get_one_string("Integrator", "path");
    }

    if (!samples_per_pixel.has_value()) {
        samples_per_pixel = 4;
    }

    const std::string sampler_type = "stratified";
    // const std::string sampler_type = "independent";

    if (sampler_type == "stratified" && integrator_name->find("mlt") == std::string::npos) {
        // MLT integrator ues it's own sampler
        const auto sqrt_val = int(std::sqrt(samples_per_pixel.value()));
        samples_per_pixel = sqr(sqrt_val);
    }

    if (integrator_name == "volpath") {
        integrator_name = "path";
    }

    if (integrator_name == "mlt" || integrator_name == "mltbdpt") {
        mlt_bdpt_integrator = MLTBDPTIntegrator::create(samples_per_pixel.value(), parameters,
                                                        integrator_base, allocator);
        return;
    }

    if (integrator_name == "mltpath") {
        mlt_path_integrator = MLTPathIntegrator::create(samples_per_pixel.value(), parameters,
                                                        integrator_base, allocator);
        return;
    }

    printf("sampler: %s\n", sampler_type.c_str());

    if (integrator_name == "bdpt") {
        bdpt_integrator = BDPTIntegrator::create(samples_per_pixel.value(), sampler_type,
                                                 parameters, integrator_base, allocator);
        return;
    }

    if (integrator_name == "path") {
        wavefront_path_integrator = WavefrontPathIntegrator::create(
            samples_per_pixel.value(), sampler_type, parameters, integrator_base, allocator);
        return;
    }

    megakernel_integrator = MegakernelIntegrator::create(integrator_name.value(), parameters,
                                                         integrator_base, allocator);
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

    if (keyword == "PixelFilter") {
        pixel_filter_tokens = tokens;
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
        const auto parameters = build_parameter_dictionary(sub_vector(tokens, 2));

        if (!samples_per_pixel.has_value()) {
            samples_per_pixel = parameters.get_integer("pixelsamples", 4);
        }

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

    if (keyword == "MakeNamedMedium" || keyword == "MediumInterface") {

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

    auto light = Light::create(light_source_type, get_render_from_object(), parameters, allocator);
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

    materials[material_name] = Material::create(type_of_material, parameters, allocator);
}

void SceneBuilder::parse_material(const std::vector<Token> &tokens) {
    if (tokens[0] != Token(TokenType::Keyword, "Material")) {
        REPORT_FATAL_ERROR();
    }

    auto type_of_material = tokens[1].values[0];

    const auto parameters = build_parameter_dictionary(sub_vector(tokens, 2));

    graphics_state.material = Material::create(type_of_material, parameters, allocator);
}

void SceneBuilder::parse_named_material(const std::vector<Token> &tokens) {
    if (tokens[0] != Token(TokenType::Keyword, "NamedMaterial")) {
        REPORT_FATAL_ERROR();
    }

    const auto material_name = tokens[1].values[0];

    if (materials.find(material_name) == materials.end()) {
        REPORT_FATAL_ERROR();
    }

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

    auto result = Shape::create(type_of_shape, render_from_object, render_from_object.inverse(),
                                graphics_state.reverse_orientation, parameters, allocator);
    auto shapes = result.first;
    auto num_shapes = result.second;

    if (!graphics_state.area_light_entity) {
        auto simple_primitives = Primitive::create_simple_primitives(
            shapes, graphics_state.material, num_shapes, allocator);

        if (active_instance_definition) {
            for (uint idx = 0; idx < num_shapes; ++idx) {
                active_instance_definition->add_primitive(&simple_primitives[idx]);
            }
        } else {
            for (uint idx = 0; idx < num_shapes; ++idx) {
                gpu_primitives.push_back(&simple_primitives[idx]);
            }
        }

        return;
    }

    // otherwise: build AreaDiffuseLight

    if (active_instance_definition) {
        printf("\nERROR: area lights not supported with object instancing\n");
        REPORT_FATAL_ERROR();
    }

    auto diffuse_area_lights =
        Light::create_diffuse_area_lights(shapes, num_shapes, render_from_object,
                                          graphics_state.area_light_entity->parameters, allocator);

    auto geometric_primitives = Primitive::create_geometric_primitives(
        shapes, graphics_state.material, diffuse_area_lights, num_shapes, allocator);

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
        auto float_texture =
            FloatTexture::create(texture_type, get_render_from_object(), parameters, allocator);
        float_textures[texture_name] = float_texture;

        return;
    }

    if (color_type == "spectrum") {
        albedo_spectrum_textures[texture_name] =
            SpectrumTexture::create(texture_type, SpectrumType::Albedo, get_render_from_object(),
                                    global_spectra->rgb_color_space, parameters, allocator);

        illuminant_spectrum_textures[texture_name] = SpectrumTexture::create(
            texture_type, SpectrumType::Illuminant, get_render_from_object(),
            global_spectra->rgb_color_space, parameters, allocator);

        unbounded_spectrum_textures[texture_name] =
            SpectrumTexture::create(texture_type, SpectrumType::Unbounded, get_render_from_object(),
                                    global_spectra->rgb_color_space, parameters, allocator);

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

    const auto transform_matrix = SquareMatrix<4>(transform_data);

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
    uint last_token_idx = 0;
    auto last_time_check = std::chrono::system_clock::now();

    auto report_time = [&token_idx, &last_token_idx, &last_time_check, &tokens,
                        function_name = __func__] {
        const auto current = std::chrono::system_clock::now();

        const std::chrono::duration<FloatType> duration{current - last_time_check};

        const auto time_in_second = duration.count();
        if (time_in_second > 5) {
            // when parsing one token took too long
            std::stringstream stream;
            stream << tokens[last_token_idx];
            const auto keyword = stream.str();

            printf("%sSceneBuilder::%s(): parsing token `%s` took %.1f seconds%s:\n",
                   FLAG_COLORFUL_PRINT_RED_START, function_name, keyword.c_str(), time_in_second,
                   FLAG_COLORFUL_PRINT_END);
            for (uint idx = last_token_idx; idx < token_idx; ++idx) {
                std::cout << tokens[idx] << "\n";
            }
        }

        last_time_check = current;
        last_token_idx = token_idx;
    };

    while (token_idx < tokens.size()) {
        if (token_idx > 0) {
            report_time();
        }

        const Token &first_token = tokens[token_idx];

        if (first_token.type == TokenType::WorldBegin) {
            build_filter();
            build_film();
            build_camera();

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

            const auto name = active_instance_definition->name;

            active_instance_definition->build_bvh(allocator);
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
                printf("\nERROR: ObjectInstance `%s` not found\n", object_name.c_str());
                REPORT_FATAL_ERROR();
            }

            const auto instance = instance_definition.at(object_name);
            if (instance->empty()) {
                token_idx += 1;
                continue;
            }

            auto world_from_render = render_from_world.inverse();
            auto render_from_instance = get_render_from_object() * world_from_render;

            const auto instanced_primitive = instance->get_instanced_primitive();

            if (render_from_instance.is_identity()) {
                gpu_primitives.push_back(instanced_primitive);
            } else {
                auto transformed_primitive = allocator.allocate<TransformedPrimitive>();
                auto primitive = allocator.allocate<Primitive>();

                transformed_primitive->init(instanced_primitive, render_from_instance);
                primitive->init(transformed_primitive);

                gpu_primitives.push_back(primitive);
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

    report_time();
}

void SceneBuilder::parse_file(const std::string &_filename) {
    const auto start = std::chrono::system_clock::now();

    const auto all_tokens = parse_pbrt_into_token(_filename);
    parse_tokens(all_tokens);

    const std::chrono::duration<FloatType> duration{std::chrono::system_clock::now() - start};

    const auto time_in_seconds = duration.count();
    if (time_in_seconds > 10) {
        // report abnormal behavior
        printf("%sSceneBuilder::%s(): took %.1f seconds%s\n", FLAG_COLORFUL_PRINT_RED_START,
               __func__, time_in_seconds, FLAG_COLORFUL_PRINT_END);
    }
}

void SceneBuilder::preprocess() {
    integrator_base->bvh = HLBVH::create(gpu_primitives, "for ROOT", allocator);

    const auto full_scene_bounds = integrator_base->bvh->bounds();
    for (auto light : gpu_lights) {
        light->preprocess(full_scene_bounds);
    }

    build_integrator();

    if (bdpt_integrator != nullptr) {
        printf("Integrator: (wavefront) bdpt\n");

    } else if (mlt_bdpt_integrator != nullptr) {
        printf("Integrator: (wavefront) mlt-bdpt\n");

    } else if (mlt_path_integrator != nullptr) {
        printf("Integrator: (wavefront) mlt-path\n");

    } else if (wavefront_path_integrator != nullptr) {
        printf("Integrator: (wavefront) path\n");

    } else if (megakernel_integrator != nullptr) {
        printf("Integrator: (megakernel) %s\n", megakernel_integrator->get_name().c_str());

    } else {
        REPORT_FATAL_ERROR();
    }
    printf("\n");

    const auto light_type_counter = count_light_type(gpu_lights);

    const auto light_size = gpu_lights.size();
    printf("total lights: %zu\n", light_size);
    for (auto const &kv : light_type_counter) {
        printf("    %s: %d (%.2f%)\n", kv.first.c_str(), kv.second,
               static_cast<double>(kv.second) / light_size * 100);
    }
    printf("\n");

    const auto material_type_counter = count_material_type();
    uint total_material_size = 0;
    for (auto const &[_, _num] : material_type_counter) {
        total_material_size += _num;
    }

    printf("total material: %zu\n", material_type_counter.size());
    for (const auto &[name, num] : material_type_counter) {
        printf("    %s: %d (%.2f%)\n", name.c_str(), num,
               static_cast<double>(num) / total_material_size * 100);
    }
    printf("\n");
}

std::map<std::string, uint> SceneBuilder::count_material_type() const {
    std::map<std::string, uint> counter;

    for (const auto primitive : gpu_primitives) {
        primitive->record_material(counter);
    }

    return counter;
}

void SceneBuilder::render() const {
    if (!integrator_base->is_ready()) {
        REPORT_FATAL_ERROR();
    }

    if (!samples_per_pixel.has_value()) {
        REPORT_FATAL_ERROR();
    }

    if (film == nullptr) {
        REPORT_FATAL_ERROR();
    }

    const auto film_resolution = film->get_resolution();

    std::string sampler_type = "stratified";
    // TODO: configure sampler_type

    auto start = std::chrono::system_clock::now();

    std::cout << "rendering a " << film_resolution.x << "x" << film_resolution.y << " image";

    const auto spp = samples_per_pixel.value();

    if (bdpt_integrator != nullptr) {
        std::cout << " (samples per pixel: " << spp << ")"
                  << " with BDPT\n"
                  << std::flush;

        auto splat_scale = 1.0 / spp;

        bdpt_integrator->render(film, spp, preview);

        film->write_to_png(output_filename, splat_scale);

    } else if (mlt_bdpt_integrator != nullptr) {
        std::cout << " (mutations per pixel: " << spp << ")"
                  << " with MLT-BDPT\n"
                  << std::flush;

        GreyScaleFilm heatmap(film_resolution);
        const auto brightness = mlt_bdpt_integrator->render(film, heatmap, spp, preview);

        film->write_to_png(output_filename, brightness / spp);

        heatmap.write_to_png("heatmap-" + output_filename);

    } else if (mlt_path_integrator != nullptr) {
        std::cout << " (mutations per pixel: " << spp << ")"
                  << " with MLT-path\n"
                  << std::flush;

        GreyScaleFilm heatmap(film_resolution);

        const auto brightness = mlt_path_integrator->render(film, heatmap, spp, preview);

        film->write_to_png(output_filename, brightness / spp);

        heatmap.write_to_png("heatmap-" + output_filename);

    } else if (wavefront_path_integrator != nullptr) {
        std::cout << " (samples per pixel: " << spp << ")"
                  << " with wavefront-path\n"
                  << std::flush;

        wavefront_path_integrator->render(film, preview);

        film->write_to_png(output_filename);

    } else if (megakernel_integrator != nullptr) {
        std::cout << " (samples per pixel: " << spp << ")"
                  << " with " + megakernel_integrator->get_name() << "\n"
                  << std::flush;

        megakernel_integrator->render(film, sampler_type, samples_per_pixel.value(),
                                      integrator_base, preview);

        film->write_to_png(output_filename);

    } else {
        REPORT_FATAL_ERROR();
    }

    const std::chrono::duration<FloatType> duration{std::chrono::system_clock::now() - start};

    std::cout << std::fixed << std::setprecision(1) << "rendering took " << duration.count()
              << " seconds\n"
              << std::flush;

    printf("GPU memory used: %s\n", allocator.get_allocated_memory_size().c_str());

    std::cout << "image saved to `" << output_filename << "`\n";
}
