#include "pbrt/base/camera.h"
#include "pbrt/cameras/perspective.h"

void Camera::init(PerspectiveCamera *perspective_camera) {
    camera_ptr = perspective_camera;
    camera_type = CameraType::perspective;
}

PBRT_CPU_GPU const CameraBase *Camera::get_camerabase() const {
    switch (camera_type) {
    case (CameraType::perspective): {
        return &((PerspectiveCamera *)camera_ptr)->camera_base;
    }
    }

    report_function_error_and_exit(__func__);
    return nullptr;
}

PBRT_CPU_GPU CameraRay Camera::generate_ray(const CameraSample &sample) const {
    switch (camera_type) {
    case (CameraType::perspective): {
        return ((PerspectiveCamera *)camera_ptr)->generate_ray(sample);
    }
    }

    report_function_error_and_exit(__func__);
    return {};
}
