#include <pbrt/base/camera.h>
#include <pbrt/gpu/gpu_memory_allocator.h>

Camera *Camera::create_perspective_camera(const Point2i &resolution,
                                          const CameraTransform &camera_transform, const Film *film,
                                          const Filter *filter, const Medium *medium,
                                          const ParameterDictionary &parameters,
                                          GPUMemoryAllocator &allocator) {
    auto camera = allocator.allocate<Camera>();
    *camera = PerspectiveCamera(resolution, camera_transform, film, filter, medium, parameters);

    return camera;
}
