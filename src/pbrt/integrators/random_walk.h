#pragma once

#include "pbrt/base/ray.h"
#include "pbrt/base/spectrum.h"
#include "pbrt/accelerator/hlbvh.h"
#include "pbrt/util/sampling.h"
#include "pbrt/euclidean_space/frame.h"

class Camera;
class Sampler;

class RandomWalkIntegrator {
  public:
    void init(const Camera *_camera, uint _max_depth);

    PBRT_GPU SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda, const HLBVH *bvh,
                                Sampler *sampler) const {
        return li_random_walk(ray, lambda, bvh, sampler, 0);
    }

  private:
    PBRT_GPU
    SampledSpectrum li_random_walk(const Ray &ray, SampledWavelengths &lambda, const HLBVH *bvh,
                                   Sampler *sampler, uint depth) const;

    uint max_depth;
    const Camera *camera;
};
