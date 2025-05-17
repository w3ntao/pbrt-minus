#include <pbrt/base/film.h>
#include <pbrt/base/filter.h>
#include <pbrt/base/sampler.h>
#include <pbrt/cameras/perspective.h>
#include <pbrt/scene/parameter_dictionary.h>

PerspectiveCamera::PerspectiveCamera(const Point2i &resolution,
                                     const CameraTransform &camera_transform, const Film *_film,
                                     const Filter *filter, const ParameterDictionary &parameters) {
    film = _film;
    camera_base.init(resolution, camera_transform);

    focal_distance = parameters.get_float("focaldistance", 1e6);
    lens_radius = parameters.get_float("lensradius", 0.0);

    auto fov = parameters.get_float("fov", 90.0);

    auto frame_aspect_ratio = Real(resolution.x) / Real(resolution.y);

    auto screen_window =
        frame_aspect_ratio > 1.0
            ? Bounds2f(Point2f(-frame_aspect_ratio, -1.0), Point2f(frame_aspect_ratio, 1.0))
            : Bounds2f(Point2f(-1.0, -1.0 / frame_aspect_ratio),
                       Point2f(1.0, 1.0 / frame_aspect_ratio));

    auto ndc_from_screen =
        Transform::scale(1.0 / (screen_window.p_max.x - screen_window.p_min.x),
                         1.0 / (screen_window.p_max.y - screen_window.p_min.y), 1.0) *
        Transform::translate(-screen_window.p_min.x, -screen_window.p_max.y, 0.0);

    auto raster_from_ndc = Transform::scale(resolution.x, -resolution.y, 1.0);

    raster_from_screen = raster_from_ndc * ndc_from_screen;

    screen_from_raster = raster_from_screen.inverse();

    screen_from_camera = Transform::perspective(fov, 1e-2, 1000.0);

    camera_from_raster = screen_from_camera.inverse() * screen_from_raster;

    dx_camera =
        camera_from_raster(Point3f(1.0, 0.0, 0.0)) - camera_from_raster(Point3f(0.0, 0.0, 0.0));

    dy_camera =
        camera_from_raster(Point3f(0.0, 1.0, 0.0)) - camera_from_raster(Point3f(0.0, 0.0, 0.0));

    // Compute _cosTotalWidth_ for perspective camera
    auto radius = Point2f(filter->get_radius());
    Point3f pCorner(-radius.x, -radius.y, 0.f);

    Vector3f wCornerCamera = this->camera_from_raster(pCorner).to_vector3().normalize();
    cosTotalWidth = wCornerCamera.z;

    // Compute image plane area at $z=1$ for _PerspectiveCamera_
    Point2i res = film->get_resolution();
    Point3f pMin = camera_from_raster(Point3f(0, 0, 0));
    Point3f pMax = camera_from_raster(Point3f(res.x, res.y, 0));
    pMin /= pMin.z;
    pMax /= pMax.z;

    A = std::abs((pMax.x - pMin.x) * (pMax.y - pMin.y));
}

PBRT_CPU_GPU
CameraRay PerspectiveCamera::generate_ray(const CameraSample &sample, Sampler *sampler) const {
    Point3f p_film = Point3f(sample.p_film.x, sample.p_film.y, 0);
    Point3f p_camera = camera_from_raster(p_film);

    Ray ray(Point3f(0, 0, 0), p_camera.to_vector3().normalize());

    if (lens_radius > 0) {
        // Sample point on lens
        Point2f pLens = lens_radius * sample_uniform_disk_concentric(sampler->get_2d());

        // Compute point on plane of focus
        auto ft = focal_distance / ray.d.z;
        Point3f pFocus = ray.at(ft);

        // Update ray for effect of lens
        ray.o = Point3f(pLens.x, pLens.y, 0);
        ray.d = (pFocus - ray.o).normalize();
    }

    return CameraRay(camera_base.camera_transform.render_from_camera(ray));
}

PBRT_CPU_GPU
void PerspectiveCamera::pdf_we(const Ray &ray, Real *pdfPos, Real *pdfDir) const {
    // Return zero PDF values if ray direction is not front-facing
    auto cosTheta =
        ray.d.dot(this->camera_base.camera_transform.render_from_camera(Vector3f(0, 0, 1)));
    if (cosTheta <= cosTotalWidth) {
        *pdfPos = *pdfDir = 0;
        return;
    }

    // Map ray $(\p{}, \w{})$ onto the raster grid
    Point3f pFocus = ray.at((lens_radius > 0 ? focal_distance : 1) / cosTheta);
    Point3f pCamera = camera_base.camera_transform.camera_from_render(pFocus);
    Point3f pRaster = camera_from_raster.apply_inverse(pCamera);

    // Return zero probability for out of bounds points
    Bounds2f sampleBounds = film->sample_bounds();
    if (!sampleBounds.contain(Point2f(pRaster.x, pRaster.y))) {
        *pdfPos = *pdfDir = 0;
        return;
    }

    // Compute lens area  and return perspective camera probabilities
    Real lensArea = lens_radius != 0 ? (compute_pi() * sqr(lens_radius)) : 1;
    *pdfPos = 1 / lensArea;
    *pdfDir = 1 / (A * pbrt::pow<3>(cosTheta));
}

PBRT_CPU_GPU
SampledSpectrum PerspectiveCamera::we(const Ray &ray, SampledWavelengths &lambda,
                                      Point2f *pRasterOut) const {
    // Check if ray is forward-facing with respect to the camera
    auto cosTheta = ray.d.dot(camera_base.camera_transform.render_from_camera(Vector3f(0, 0, 1)));
    if (cosTheta <= cosTotalWidth) {
        return SampledSpectrum(0.0);
    }

    // Map ray $(\p{}, \w{})$ onto the raster grid
    Point3f pFocus = ray.at((lens_radius > 0 ? focal_distance : 1) / cosTheta);
    Point3f pCamera = camera_base.camera_transform.camera_from_render(pFocus);
    Point3f pRaster = camera_from_raster.apply_inverse(pCamera);

    if (pRasterOut != nullptr) {
        // Return raster position if requested
        *pRasterOut = Point2f(pRaster.x, pRaster.y);
    }

    // Return zero importance for out of bounds points
    Bounds2f sampleBounds = film->sample_bounds();
    if (!sampleBounds.contain(Point2f(pRaster.x, pRaster.y))) {
        return SampledSpectrum(0.0);
    }

    // Compute lens area of perspective camera
    auto lensArea = lens_radius != 0 ? (compute_pi() * sqr(lens_radius)) : 1;

    // Return importance for point on image plane
    return SampledSpectrum(1.0 / (A * lensArea * pbrt::pow<4>(cosTheta)));
}

PBRT_CPU_GPU
pbrt::optional<CameraWiSample> PerspectiveCamera::sample_wi(const Interaction &ref, const Point2f u,
                                                            SampledWavelengths &lambda) const {
    // Uniformly sample a lens interaction _lensIntr_
    Point2f pLens = lens_radius * sample_uniform_disk_concentric(u);
    auto pLensRender =
        camera_base.camera_transform.render_from_camera(Point3f(pLens.x, pLens.y, 0));
    auto n = Normal3f(camera_base.camera_transform.render_from_camera(Vector3f(0, 0, 1)));
    Interaction lensIntr(pLensRender, n);

    // Find incident direction to camera _wi_ at _ref_
    Vector3f wi = lensIntr.p() - ref.p();
    auto dist = wi.length();
    wi /= dist;

    // Compute PDF for importance arriving at _ref_
    auto lensArea = lens_radius != 0 ? (compute_pi() * sqr(lens_radius)) : 1;
    auto pdf = sqr(dist) / (lensIntr.n.abs_dot(wi) * lensArea);

    // Compute importance and return _CameraWiSample_
    Point2f pRaster;
    SampledSpectrum spectrum_wi = we(lensIntr.spawn_ray(-wi), lambda, &pRaster);

    if (!spectrum_wi.is_positive()) {
        return {};
    }

    return CameraWiSample(spectrum_wi, wi, pdf, pRaster, ref, lensIntr);
}
