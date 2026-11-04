#pragma once

#include <memory>
#include <string>

class Film;
class GreyScaleFilm;
class MLTBDPTIntegrator;
class ParameterDictionary;
class Spectrum;
struct IntegratorBase;

// Pixel-selective PT / MLT-BDPT. The independently randomized pilot
// only trains a frozen mask; it never contributes samples to the final image.
class SelectiveMLTIntegrator {
  public:
    SelectiveMLTIntegrator(int samples_per_pixel, const ParameterDictionary &parameters, const IntegratorBase *base,
                           const Film *film);
    ~SelectiveMLTIntegrator();

    // PT is normalized by Film's sample weights. The return value scales only
    // the raw MLT splats, so the two estimators compose in linear sensor space.
    double render(Film *film, GreyScaleFilm &firefly_mask, GreyScaleFilm &density,
                  const std::string &output_filename) const;

  private:
    const IntegratorBase *base;
    const Spectrum *cie_y;
    int samples_per_pixel;
    int pilot_samples_per_pixel;
    int mutations_per_pixel;
    int bootstrap_samples;
    int num_chains;
    int max_depth;
    bool regularize;
    std::unique_ptr<MLTBDPTIntegrator> mlt;
};
