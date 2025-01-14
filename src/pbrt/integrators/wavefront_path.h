#pragma once

#include <pbrt/base/material.h>
#include <pbrt/euclidean_space/point2.h>

class Film;
class GPUMemoryAllocator;
class ParameterDictionary;
class Sampler;
class SampledSpectrum;
class SampledWavelengths;
class Spectrum;
class SurfaceInteraction;

class BSDF;

struct CameraSample;
struct CameraRay;
struct FrameBuffer;
struct MISParameter;
struct IntegratorBase;
struct ShapeIntersection;

class WavefrontPathIntegrator {
  public:
    struct PathState {
        CameraSample *camera_samples;
        CameraRay *camera_rays;
        SampledWavelengths *lambdas;

        SampledSpectrum *L;
        SampledSpectrum *beta;
        Sampler *samplers;

        pbrt::optional<ShapeIntersection> *shape_intersections;

        uint *path_length;
        bool *finished;

        BSDF *bsdf;

        MISParameter *mis_parameters;

        uint *pixel_indices;
        uint *sample_indices;

        Point2i image_resolution;

        unsigned long long int global_path_counter;
        unsigned long long int total_path_num;

        void create(uint samples_per_pixel, const Point2i &_resolution,
                    const std::string &sampler_type, GPUMemoryAllocator &allocator);

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

        void init(GPUMemoryAllocator &allocator);
    };

    static WavefrontPathIntegrator *create(uint samples_per_pixel, const std::string &sampler_type,
                                           const ParameterDictionary &parameters,
                                           const IntegratorBase *base,
                                           GPUMemoryAllocator &allocator);

    void render(Film *film, bool preview);

    PathState path_state;

    Queues queues;

    const IntegratorBase *base;
    uint max_depth;
    bool regularize;
    uint samples_per_pixel;

    PBRT_CPU_GPU
    SampledSpectrum sample_ld(const SurfaceInteraction &intr, const BSDF *bsdf,
                              SampledWavelengths &lambda, Sampler *sampler) const;

    PBRT_CPU_GPU
    void sample_bsdf(uint path_idx, PathState *path_state) const;

    void evaluate_material(const Material::Type material_type);
};
