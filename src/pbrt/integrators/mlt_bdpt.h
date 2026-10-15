#pragma once

#include <pbrt/base/spectrum.h>
#include <vector>

class Film;
class GPUMemoryAllocator;
class GreyScaleFilm;
class ParameterDictionary;

class MLTSampler;
class Sampler;

struct BDPTConfig;
struct FilmSample;
struct IntegratorBase;
struct Vertex;

struct BDPTPathSample {
    Point2f p_film;
    SampledSpectrum radiance;
    SampledWavelengths lambda;
    Real depth;

    static BDPTPathSample zero(Real depth) {
        return BDPTPathSample(Point2f(NAN, NAN), 0, SampledWavelengths::zero(), depth);
    }

    BDPTPathSample(const Point2f &_p_film, const SampledSpectrum &_radiance, const SampledWavelengths &_lambda,
                   Real _depth)
        : p_film(_p_film), radiance(_radiance), lambda(_lambda), depth(_depth) {}
};

class MLTBDPTIntegrator {
  public:
    static constexpr int cameraStreamIndex = 0;
    static constexpr int lightStreamIndex = 1;
    static constexpr int connectionStreamIndex = 2;
    static constexpr int nSampleStreams = 3;

    MLTBDPTIntegrator(const BDPTConfig *_config, const Point2i &_film_dimension, const Spectrum *_cie_y,
                      int _mutations_per_pixel, int _num_samplers, Real _sigma, Real _large_step_probability)
        : config(_config), film_dimension(_film_dimension), cie_y(_cie_y), mutations_per_pixel(_mutations_per_pixel),
          num_samplers(_num_samplers), sigma(_sigma), large_step_probability(_large_step_probability) {}

    static MLTBDPTIntegrator *create(int mutations_per_pixel, const ParameterDictionary &parameters,
                                     const IntegratorBase *base, GPUMemoryAllocator &allocator);

    double render(Film *film, GreyScaleFilm &heat_map, int mutations_per_pixel);

    BDPTPathSample li(int depth, Sampler *sampler) const;

    Real compute_importance(const BDPTPathSample &path) const;

    const BDPTConfig *config = nullptr;

  private:
    BDPTPathSample generate_path_sample(MLTSampler &mlt_sampler) const;
    BDPTPathSample generate_bootstrap_path(long bootstrap_index, MLTSampler &mlt_sampler) const;

    void build_bootstrap_samples(int worker_idx, int num_paths_per_worker,
                                 std::vector<double> &importance_per_path) const;
    void mutate_markov_chain(int worker_idx, Film *sub_film, GreyScaleFilm &sub_heat_map, long long num_mutations,
                             const std::vector<double> &importance_per_path,
                             const std::vector<double> &bootstrap_cdf) const;

    Point2i film_dimension = {0, 0};

    const Spectrum *cie_y = nullptr;

    int mutations_per_pixel = 0;
    int num_samplers = 0;
    Real sigma = 0.01;
    Real large_step_probability = 0.3;
};
