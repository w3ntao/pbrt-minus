#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/textures/gpu_image.h>
#include <pbrt/textures/mipmap.h>

const MIPMap *MIPMap::create(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator) {
    auto mipmap = allocator.allocate<MIPMap>();
    *mipmap = MIPMap(parameters, allocator);

    return mipmap;
}

MIPMap::MIPMap(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator) {
    auto max_anisotropy = parameters.get_float("maxanisotropy", 8.0);
    auto filter_string = parameters.get_one_string("filter", "bilinear");

    options = MIPMapFilterOptions{
        .filter = parse_filter_function(filter_string),
        .max_anisotropy = max_anisotropy,
    };

    auto wrap_string = parameters.get_one_string("wrap", "repeat");
    wrap_mode = parse_wrap_mode(wrap_string);

    auto image_path = parameters.root + "/" + parameters.get_one_string("filename");
    image = GPUImage::create_from_file(image_path, allocator);
}

PBRT_CPU_GPU
RGB MIPMap::filter(const Point2f st) const {
    return image->bilerp(st, WrapMode2D(wrap_mode));
}
