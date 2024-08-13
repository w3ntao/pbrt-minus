#include "pbrt/base/camera.h"
#include "pbrt/cameras/perspective.h"

Camera *Camera::create_perspective_camera(const Point2i &resolution,
                                          const CameraTransform &camera_transform, FloatType fov,
                                          FloatType lens_radius,
                                          std::vector<void *> &gpu_dynamic_pointers) {
    PerspectiveCamera *perspective_camera;
    CHECK_CUDA_ERROR(cudaMallocManaged(&perspective_camera, sizeof(PerspectiveCamera)));

    Camera *camera;
    CHECK_CUDA_ERROR(cudaMallocManaged(&camera, sizeof(Camera)));

    gpu_dynamic_pointers.push_back(perspective_camera);
    gpu_dynamic_pointers.push_back(camera);

    perspective_camera->init(resolution, camera_transform, fov, lens_radius);
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
    case (Type::perspective): {
        return &((PerspectiveCamera *)ptr)->camera_base;
    }
    }

    REPORT_FATAL_ERROR();
    return nullptr;
}

PBRT_CPU_GPU
CameraRay Camera::generate_ray(const CameraSample &sample) const {
    switch (type) {
    case (Type::perspective): {
        return ((PerspectiveCamera *)ptr)->generate_ray(sample);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
