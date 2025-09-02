#pragma once

#include <pbrt/base/material.h>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/gpu/gpu_memory_allocator.h>

class Film;
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
    static constexpr int PATH_POOL_SIZE = 1 * 1024 * 1024;

    struct PathState {
        CameraSample *camera_samples = nullptr;
        CameraRay *camera_rays = nullptr;
        SampledWavelengths *lambdas = nullptr;

        SampledSpectrum *L = nullptr;
        SampledSpectrum *beta = nullptr;
        Sampler *samplers = nullptr;

        pbrt::optional<ShapeIntersection> *shape_intersections = nullptr;

        Real *path_length = nullptr;
        // making depth float type to counter for contribution from material-less hit
        bool *finished = nullptr;

        BSDF *bsdf = nullptr;

        MISParameter *mis_parameters = nullptr;

        int *pixel_indices = nullptr;
        int *sample_indices = nullptr;

        const Point2i image_resolution = {0, 0};
        const int samples_per_pixel;
        const std::string sampler_type;
        const unsigned long long int total_path_num = 0;

        unsigned long long int global_path_counter = 0;

        PathState(const Point2i &_resolution, int _samples_per_pixel,
                  const std::string &sampler_type);

        void build_path(GPUMemoryAllocator &allocator);

        PBRT_CPU_GPU
        void reset_path(int path_idx);
    };

    struct Queues {
        struct SingleQueue {
            int *queue_array = nullptr;
            int counter = 0;

            explicit SingleQueue(GPUMemoryAllocator &allocator)
                : queue_array(allocator.allocate<int>(PATH_POOL_SIZE)) {}

            PBRT_GPU
            void append_path(const int path_idx) {
                const int queue_idx = atomicAdd(&counter, 1);
                queue_array[queue_idx] = path_idx;
            }
        };

        explicit Queues(GPUMemoryAllocator &allocator);

        SingleQueue *new_paths = nullptr;
        SingleQueue *rays = nullptr;

        SingleQueue *conductor_material = nullptr;
        SingleQueue *coated_conductor_material = nullptr;
        SingleQueue *coated_diffuse_material = nullptr;
        SingleQueue *dielectric_material = nullptr;
        SingleQueue *diffuse_material = nullptr;
        SingleQueue *diffuse_transmission_material = nullptr;

        int frame_buffer_counter = 0;
        FrameBuffer *frame_buffer_queue = nullptr;

        [[nodiscard]]
        std::vector<SingleQueue *> get_all_queues() const {
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
    };

    WavefrontPathIntegrator(int _samples_per_pixel, const std::string &sampler_type,
                            const ParameterDictionary &parameters, const IntegratorBase *_base,
                            GPUMemoryAllocator &allocator);

    void render(Film *film, bool preview);

    PathState path_state;

    Queues queues;

    const IntegratorBase *base = nullptr;
    const int samples_per_pixel = 0;

    const int max_depth = 0;
    const bool regularize = false;

    PBRT_CPU_GPU
    void sample_bsdf(int path_idx, PathState *path_state) const;

    void evaluate_material(Material::Type material_type);
};
