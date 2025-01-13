#include <pbrt/base/camera.h>
#include <pbrt/cameras/perspective.h>
#include <pbrt/gpu/gpu_memory_allocator.h>

Camera *Camera::create_perspective_camera(const Point2i &resolution,
                                          const CameraTransform &camera_transform, const Film *film,
                                          const Filter *filter,
                                          const ParameterDictionary &parameters,
                                          GPUMemoryAllocator &allocator) {
    auto perspective_camera = allocator.allocate<PerspectiveCamera>();
    auto camera = allocator.allocate<Camera>();

    perspective_camera->init(resolution, camera_transform, film, filter, parameters);
    camera->init(perspective_camera);

    return camera;
}

void Camera::init(const PerspectiveCamera *perspective_camera) {
    ptr = perspective_camera;
    type = Type::perspective;
}

PBRT_CPU_GPU
const CameraBase *Camera::get_camerabase() const {
    switch (type) {
    case Type::perspective: {
        return &(static_cast<const PerspectiveCamera *>(ptr)->camera_base);
    }
    }

    REPORT_FATAL_ERROR();
    return nullptr;
}

PBRT_CPU_GPU
CameraRay Camera::generate_ray(const CameraSample &sample, Sampler *sampler) const {
    switch (type) {
    case Type::perspective: {
        return static_cast<const PerspectiveCamera *>(ptr)->generate_ray(sample, sampler);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
void Camera::pdf_we(const Ray &ray, FloatType *pdfPos, FloatType *pdfDir) const {
    switch (type) {
    case Type::perspective: {
        return static_cast<const PerspectiveCamera *>(ptr)->pdf_we(ray, pdfPos, pdfDir);
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_CPU_GPU
pbrt::optional<CameraWiSample> Camera::sample_wi(const Interaction &ref, const Point2f u,
                                                 SampledWavelengths &lambda) const {
    switch (type) {
    case Type::perspective: {
        return static_cast<const PerspectiveCamera *>(ptr)->sample_wi(ref, u, lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
