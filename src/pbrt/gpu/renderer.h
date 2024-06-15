#pragma once

#include <iostream>
#include <string>

#include "pbrt/base/camera.h"
#include "pbrt/base/filter.h"
#include "pbrt/base/integrator.h"
#include "pbrt/base/sampler.h"
#include "pbrt/base/shape.h"
#include "pbrt/base/spectrum.h"

#include "pbrt/accelerator/hlbvh.h"

#include "pbrt/cameras/perspective.h"

#include "pbrt/films/pixel_sensor.h"
#include "pbrt/films/rgb_film.h"

#include "pbrt/lights/diffuse_area_light.h"

#include "pbrt/primitives/simple_primitives.h"
#include "pbrt/primitives/geometric_primitive.h"

#include "pbrt/samplers/independent_sampler.h"

#include "pbrt/shapes/triangle.h"

#include "pbrt/spectrum_util/color_encoding.h"
#include "pbrt/spectrum_util/global_spectra.h"
#include "pbrt/spectrum_util/rgb_color_space.h"
#include "pbrt/spectrum_util/sampled_wavelengths.h"
#include "pbrt/spectrum_util/spectrum_constants_cie.h"

class Renderer {
  public:
    Integrator *integrator;
    Camera *camera;
    Filter *filter;
    Film *film;
    HLBVH *bvh;
    Sampler *samplers;

    PixelSensor sensor;

    PBRT_GPU
    void evaluate_pixel_sample(const Point2i p_pixel, const int num_samples) {
        int width = camera->get_camerabase()->resolution.x;
        const uint pixel_index = p_pixel.y * width + p_pixel.x;

        auto sampler = &samplers[pixel_index];
        sampler->start_pixel_sample(pixel_index, 0, 0);

        for (uint i = 0; i < num_samples; ++i) {
            auto camera_sample = sampler->get_camera_sample(p_pixel, filter);
            auto lu = sampler->get_1d();
            auto lambda = SampledWavelengths::sample_visible(lu);

            auto ray = camera->generate_camera_differential_ray(camera_sample);

            auto radiance_l = ray.weight * integrator->li(ray.ray, lambda, sampler);

            if (radiance_l.has_nan()) {
                printf("%s(): pixel(%d, %d), samples %u: has NAN\n", __func__, p_pixel.x, p_pixel.y,
                       i);
            }

            film->add_sample(p_pixel, radiance_l, lambda, camera_sample.filter_weight);
        }
    }
};
