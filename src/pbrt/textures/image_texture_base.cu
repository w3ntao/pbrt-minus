#include "pbrt/euclidean_space/transform.h"
#include "pbrt/textures/image_texture_base.h"
#include "pbrt/textures/mipmap.h"

void ImageTextureBase::init_image_texture_base(const Transform &render_from_object,
                                               const ParameterDictionary &parameters,
                                               std::vector<void *> &gpu_dynamic_pointers) {
    mipmap = MIPMap::create(parameters, gpu_dynamic_pointers);

    scale = parameters.get_float("scale", 1.0);
    invert = parameters.get_bool("invert", false);

    texture_mapping = nullptr;

    const std::string mapping = parameters.get_one_string("mapping", "uv");

    if (mapping == "uv") {
        TextureMapping2D *_texture_mapping;
        UVMapping *uv_mapping;

        CHECK_CUDA_ERROR(cudaMallocManaged(&_texture_mapping, sizeof(TextureMapping2D)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&uv_mapping, sizeof(UVMapping)));

        gpu_dynamic_pointers.push_back(uv_mapping);
        gpu_dynamic_pointers.push_back(_texture_mapping);

        uv_mapping->init(parameters);
        _texture_mapping->init(uv_mapping);

        texture_mapping = _texture_mapping;

        return;
    }

    printf("\ntexture mapping `%s` not implemented\n", mapping.c_str());

    REPORT_FATAL_ERROR();
}
