#include "pbrt/base/camera.h"
#include "pbrt/cameras/perspective.h"

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

PBRT_CPU_GPU
CameraDifferentialRay Camera::generate_camera_differential_ray(const CameraSample &sample) const {
    switch (type) {
    case (Type::perspective): {
        return ((PerspectiveCamera *)ptr)->generate_camera_differential_ray(sample);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
