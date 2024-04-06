#include "pbrt/base/film.h"
#include "pbrt/films/rgb_film.h"

void Film::init(RGBFilm *rgb_film) {
    film_ptr = rgb_film;
    film_type = FilmType::rgb;
}

PBRT_CPU_GPU
void Film::add_sample(const Point2i &p_film, const SampledSpectrum &radiance_l,
                      const SampledWavelengths &lambda, double weight) {
    switch (film_type) {
    case (FilmType::rgb): {
        return ((RGBFilm *)film_ptr)->add_sample(p_film, radiance_l, lambda, weight);
    }
    }

    report_error();
}

PBRT_CPU_GPU
RGB Film::get_pixel_rgb(const Point2i &p) const {
    switch (film_type) {
    case (FilmType::rgb): {
        return ((RGBFilm *)film_ptr)->get_pixel_rgb(p);
    }
    }

    report_error();
}