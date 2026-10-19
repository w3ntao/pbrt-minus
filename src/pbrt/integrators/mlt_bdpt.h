#pragma once

#include <cstdint>
#include <memory>
#include <pbrt/base/spectrum.h>
#include <pbrt/integrators/bdpt.h>
#include <vector>

class Film;
class GreyScaleFilm;
class ParameterDictionary;

class Sampler;

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

    MLTBDPTIntegrator(const IntegratorBase *_base, int _max_depth, bool _regularize, const Point2i &_film_dimension,
                      const Spectrum *_cie_y, int _mutations_per_pixel, int _num_samplers, Real _sigma,
                      Real _large_step_probability)
        : bdpt(_base, _max_depth, _regularize), film_dimension(_film_dimension), cie_y(_cie_y),
          mutations_per_pixel(_mutations_per_pixel), num_samplers(_num_samplers), sigma(_sigma),
          large_step_probability(_large_step_probability) {}

    static std::unique_ptr<MLTBDPTIntegrator> create(int mutations_per_pixel, const ParameterDictionary &parameters,
                                                     const IntegratorBase *base);

    double render(Film *film, GreyScaleFilm &density_map, int mutations_per_pixel);

    // A frozen pixel mask selects the destination splats of this separate estimator.
    double render_firefly(Film *film, GreyScaleFilm &density_map, const uint8_t *mask, int mutations_per_pixel,
                         int bootstrap_samples, int num_chains) const;

    BDPTPathSample li(int depth, Sampler *sampler) const;

    Real compute_importance(const BDPTPathSample &path) const;

    Real compute_selective_mlt_importance(const BDPTPathSample &path, const Film *film, const uint8_t *mask) const;

    const BDPTIntegrator bdpt;

  private:
    BDPTPathSample generate_path_sample(Sampler *sampler) const;
    BDPTPathSample generate_bootstrap_path(long bootstrap_index, Sampler *sampler) const;

    void build_bootstrap_samples(int worker_idx, int num_paths_per_worker,
                                 std::vector<double> &importance_per_path) const;
    void mutate_markov_chain(int worker_idx, Film *sub_film, GreyScaleFilm &sub_density, long long num_mutations,
                             const std::vector<double> &importance_per_path,
                             const std::vector<double> &bootstrap_cdf) const;

    Point2i film_dimension = {0, 0};

    const Spectrum *cie_y = nullptr;

    int mutations_per_pixel = 0;
    int num_samplers = 0;
    Real sigma = 0.01;
    Real large_step_probability = 0.3;
};
