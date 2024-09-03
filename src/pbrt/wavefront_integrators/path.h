#pragma once

#include "pbrt/base/material.h"
#include "pbrt/euclidean_space/point2.h"
#include <vector>

class CameraSample;
class CameraRay;
class Film;
class Filter;
class FrameBuffer;
class IntegratorBase;
class MISParameter;
class ParameterDictionary;
class Sampler;
class SampledSpectrum;
class SampledWavelengths;
class ShapeIntersection;
class Spectrum;
class SurfaceInteraction;

class BSDF;
class CoatedConductorBxDF;
class CoatedDiffuseBxDF;
class ConductorBxDF;
class DielectricBxDF;
class DiffuseBxDF;

struct PathState {
    CameraSample *camera_samples;
    CameraRay *camera_rays;
    SampledWavelengths *lambdas;

    SampledSpectrum *L;
    SampledSpectrum *beta;
    Sampler *samplers;

    ShapeIntersection *shape_intersections;

    uint *path_length;
    bool *intersected;
    bool *finished;

    BSDF *bsdf;
    CoatedConductorBxDF *coated_conductor_bxdf;
    CoatedDiffuseBxDF *coated_diffuse_bxdf;
    ConductorBxDF *conductor_bxdf;
    DielectricBxDF *dielectric_bxdf;
    DiffuseBxDF *diffuse_bxdf;

    MISParameter *mis_parameters;

    uint *pixel_indices;
    uint *sample_indices;

    Point2i image_resolution;

    unsigned long long int global_path_counter;
    unsigned long long int total_path_num;

    void create(uint samples_per_pixel, const Point2i &_resolution, const std::string &sampler_type,
                std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU
    void init_new_path(uint path_idx);
};

struct Queues {
    uint *new_path_queue;
    uint new_path_counter;

    uint frame_buffer_counter;
    FrameBuffer *frame_buffer_queue;

    uint *ray_queue;
    uint ray_counter;

    uint *conductor_material_queue;
    uint conductor_material_counter;

    uint *coated_conductor_material_queue;
    uint coated_conductor_material_counter;

    uint *coated_diffuse_material_queue;
    uint coated_diffuse_material_counter;

    uint *dielectric_material_queue;
    uint dielectric_material_counter;

    uint *diffuse_material_queue;
    uint diffuse_material_counter;

    void init(std::vector<void *> &gpu_dynamic_pointers);
};

class WavefrontPathIntegrator {
  public:
    static WavefrontPathIntegrator *create(const ParameterDictionary &parameters,
                                           const IntegratorBase *base,
                                           const std::string &sampler_type, uint samples_per_pixel,
                                           std::vector<void *> &gpu_dynamic_pointers);

    void render(Film *film, const Filter *filter);

    PathState path_state;

    Queues queues;

    const IntegratorBase *base;

    uint max_depth;

    PBRT_GPU
    SampledSpectrum sample_ld(const SurfaceInteraction &intr, const BSDF *bsdf,
                              SampledWavelengths &lambda, Sampler *sampler) const;

    PBRT_GPU void sample_bsdf(uint path_idx, PathState *path_state) const;

    template <Material::Type material_type>
    void evaluate_material();
};
