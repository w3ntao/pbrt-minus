#include "pbrt/base/camera.h"
#include "pbrt/cameras/perspective.h"

void Camera::init(const PerspectiveCamera *perspective_camera) {
    camera_ptr = perspective_camera;
    camera_type = Type::perspective;
}

PBRT_CPU_GPU
const CameraBase *Camera::get_camerabase() const {
    switch (camera_type) {
    case (Type::perspective): {
        return &((PerspectiveCamera *)camera_ptr)->camera_base;
    }
    }

    REPORT_FATAL_ERROR();
    return nullptr;
}

PBRT_CPU_GPU
CameraRay Camera::generate_ray(const CameraSample &sample) const {
    switch (camera_type) {
    case (Type::perspective): {
        return ((PerspectiveCamera *)camera_ptr)->generate_ray(sample);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
CameraDifferentialRay Camera::generate_camera_differential_ray(const CameraSample &sample) const {
    switch (camera_type) {
    case (Type::perspective): {
        return ((PerspectiveCamera *)camera_ptr)->generate_camera_differential_ray(sample);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
