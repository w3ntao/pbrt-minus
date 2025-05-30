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

        int *path_length;
        bool *finished;

        BSDF *bsdf;

        MISParameter *mis_parameters;

        int *pixel_indices;
        int *sample_indices;

        Point2i image_resolution;

        unsigned long long int global_path_counter;
        unsigned long long int total_path_num;

        void create(int samples_per_pixel, const Point2i &_resolution,
                    const std::string &sampler_type, GPUMemoryAllocator &allocator);

        PBRT_CPU_GPU
        void init_new_path(int path_idx);
    };

    struct Queues {
        struct SingleQueue {
            int *queue_array;
            int counter;

            PBRT_GPU
            void append_path(const int path_idx) {
                const int queue_idx = atomicAdd(&counter, 1);
                queue_array[queue_idx] = path_idx;
            }
        };

        SingleQueue *new_paths;
        SingleQueue *rays;

        SingleQueue *conductor_material;
        SingleQueue *coated_conductor_material;
        SingleQueue *coated_diffuse_material;
        SingleQueue *dielectric_material;
        SingleQueue *diffuse_material;
        SingleQueue *diffuse_transmission_material;

        int frame_buffer_counter;
        FrameBuffer *frame_buffer_queue;

        void init(GPUMemoryAllocator &allocator);

        [[nodiscard]] std::vector<SingleQueue *> get_all_queues() const {
            auto all_queues = std::vector({new_paths, rays});

            for (const auto material_type : Material::get_basic_material_types()) {
                all_queues.push_back(get_material_queue(material_type));
            }

            return all_queues;
        }

        SingleQueue *get_material_queue(const Material::Type material_type) const {
            switch (material_type) {
            case Material::Type::coated_conductor: {
                return coated_conductor_material;
            }

            case Material::Type::coated_diffuse: {
                return coated_diffuse_material;
            }

            case Material::Type::conductor: {
                return conductor_material;
            }

            case Material::Type::dielectric: {
                return dielectric_material;
            }

            case Material::Type::diffuse: {
                return diffuse_material;
            }

            case Material::Type::diffuse_transmission: {
                return diffuse_transmission_material;
            }
            }

            REPORT_FATAL_ERROR();
            return nullptr;
        }

      private:
        static SingleQueue *build_new_queue(GPUMemoryAllocator &allocator);
    };

    static WavefrontPathIntegrator *create(int samples_per_pixel, const std::string &sampler_type,
                                           const ParameterDictionary &parameters,
                                           const IntegratorBase *base,
                                           GPUMemoryAllocator &allocator);

    void render(Film *film, bool preview);

    PathState path_state;

    Queues queues;

    const IntegratorBase *base;
    int max_depth;
    bool regularize;
    int samples_per_pixel;

    PBRT_CPU_GPU
    SampledSpectrum sample_ld(const SurfaceInteraction &intr, const BSDF *bsdf,
                              SampledWavelengths &lambda, Sampler *sampler) const;

    PBRT_CPU_GPU
    void sample_bsdf(int path_idx, PathState *path_state) const;

    void evaluate_material(const Material::Type material_type);
};
