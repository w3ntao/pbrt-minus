#pragma once

#include <cuda/std/variant>
#include <pbrt/cameras/perspective.h>

namespace HIDDEN {
using CameraVariants = cuda::std::variant<PerspectiveCamera>;
}

class Camera : public HIDDEN::CameraVariants {
    using HIDDEN::CameraVariants::CameraVariants;

  public:
    static Camera *
    create_perspective_camera(const Point2i &resolution, const CameraTransform &camera_transform,
                              const Film *film, const Filter *filter, const Medium *medium,
                              const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);
    PBRT_CPU_GPU
    const CameraBase *get_camera_base() const {
        return cuda::std::visit([&](auto &x) { return x.get_camera_base(); }, *this);
    }

    PBRT_CPU_GPU
    CameraRay generate_ray(const CameraSample &sample, Sampler *sampler) const {
        return cuda::std::visit([&](auto &x) { return x.generate_ray(sample, sampler); }, *this);
    }

    PBRT_CPU_GPU
    void approximate_dp_dxy(const Point3f p, const Normal3f n, int samples_per_pixel,
                            Vector3f *dpdx, Vector3f *dpdy) const {
        get_camera_base()->approximate_dp_dxy(p, n, samples_per_pixel, dpdx, dpdy);
    }

    PBRT_CPU_GPU
    void pdf_we(const Ray &ray, Real *pdfPos, Real *pdfDir) const {
        return cuda::std::visit([&](auto &x) { return x.pdf_we(ray, pdfPos, pdfDir); }, *this);
    }

    PBRT_CPU_GPU
    pbrt::optional<CameraWiSample> sample_wi(const Interaction &ref, const Point2f u,
                                             SampledWavelengths &lambda) const {
        return cuda::std::visit([&](auto &x) { return x.sample_wi(ref, u, lambda); }, *this);
    }
};
