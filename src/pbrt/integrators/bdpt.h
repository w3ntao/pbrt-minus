#pragma once

#include <memory>
#include <pbrt/base/light.h>
#include <pbrt/base/sampler.h>
#include <vector>

struct IntegratorBase;
struct FilmSample;

struct EndpointInteraction : Interaction {
    const Camera *camera = nullptr;
    const Light *light = nullptr;

    EndpointInteraction() {}

    EndpointInteraction(const Light *_light, const Ray &ray) : Interaction(ray.o, ray.medium), light(_light) {}

    EndpointInteraction(const Camera *_camera, const Ray &ray) : Interaction(ray.o, ray.medium), camera(_camera) {}

    EndpointInteraction(const Light *_light, const Interaction &_interaction)
        : Interaction(_interaction), light(_light) {}

    EndpointInteraction(const Interaction &_interaction, const Camera *_camera)
        : Interaction(_interaction), camera(_camera) {}

    explicit EndpointInteraction(const Ray &ray) : Interaction(ray.at(1), Normal3f(-ray.d), ray.medium) {}
};

struct Vertex {
    enum class VertexType { camera, light, surface, medium };

    VertexType type;
    EndpointInteraction ei;
    SurfaceInteraction si;
    Interaction mi;
    // TODO: rewrite ei/si/mi with union or std::variant
    BSDF bsdf;

    SampledSpectrum beta = SampledSpectrum(NAN);
    bool delta = false;
    Real pdfFwd = 0;
    Real pdfRev = 0;

    Vertex() : type(VertexType::camera), beta(NAN) {}

    Vertex(const VertexType _type, const EndpointInteraction &_ei, const SampledSpectrum &_beta)
        : type(_type), ei(_ei), beta(_beta) {}

    Vertex(const SurfaceInteraction &_si, const BSDF &_bsdf, const SampledSpectrum &_beta)
        : type(VertexType::surface), si(_si), bsdf(_bsdf), beta(_beta) {}

    Vertex(const Interaction &mi, const SampledSpectrum &beta) : type(VertexType::medium), mi(mi), beta(beta) {}

    bool is_light() const {
        return type == VertexType::light || (type == VertexType::surface && si.area_light != nullptr);
    }

    bool is_delta_light() const {
        return type == VertexType::light && ei.light && pbrt::is_delta_light(ei.light->get_light_type());
    }

    static Vertex create_camera(const Camera *camera, const Ray &ray, const SampledSpectrum &beta) {
        return Vertex(VertexType::camera, EndpointInteraction(camera, ray), beta);
    }

    static Vertex create_camera(const Camera *camera, const Interaction &it, const SampledSpectrum &beta) {
        return Vertex(VertexType::camera, EndpointInteraction(it, camera), beta);
    }

    static Vertex create_light(const EndpointInteraction &ei, const SampledSpectrum &beta, const Real pdf) {
        Vertex v(VertexType::light, ei, beta);
        v.pdfFwd = pdf;
        return v;
    }

    static Vertex create_light(const Light *light, const Interaction &intractopm, const SampledSpectrum &Le,
                               const Real pdf) {
        Vertex v(VertexType::light, EndpointInteraction(light, intractopm), Le);
        v.pdfFwd = pdf;
        return v;
    }

    static Vertex create_light(const Light *light, const Ray &ray, const SampledSpectrum &Le, const Real pdf) {
        Vertex v(VertexType::light, EndpointInteraction(light, ray), Le);
        v.pdfFwd = pdf;
        return v;
    }

    static Vertex create_surface(const SurfaceInteraction &si, const BSDF &bsdf, const SampledSpectrum &beta, Real pdf,
                                 const Vertex &prev) {
        Vertex v(si, bsdf, beta);
        v.pdfFwd = prev.convert_density(pdf, v);
        return v;
    }

    static Vertex create_medium(const Interaction &mi, const SampledSpectrum &beta, const Real pdf,
                                const Vertex &prev) {
        Vertex v(mi, beta);
        v.pdfFwd = prev.convert_density(pdf, v);
        return v;
    }

    bool is_connectible() const {
        switch (type) {
        case VertexType::medium: {
            return true;
        }

        case VertexType::light: {
            return ei.light->get_light_type() != LightType::delta_direction;
        }

        case VertexType::camera: {
            return true;
        }

        case VertexType::surface: {
            return pbrt::is_non_specular(bsdf.flags());
        }
        }

        REPORT_FATAL_ERROR();
        return false;
    }

    const Interaction &get_interaction() const {
        switch (type) {
        case VertexType::medium: {
            return mi;
        }
        case VertexType::surface: {
            return si;
        }
        default: {
            return ei;
        }
        }

        REPORT_FATAL_ERROR();
    }

    const SurfaceInteraction &get_surface_interaction() const {
        if (type == VertexType::surface) {
            return si;
        }

        REPORT_FATAL_ERROR();
    }

    Point3f p() const {
        return get_interaction().p();
    }

    const Normal3f &ng() const {
        return get_interaction().n;
    }

    const Normal3f &ns() const {
        if (type == VertexType::surface) {
            return si.shading.n;
        }

        return get_interaction().n;
    }

    bool is_on_surface() const {
        return get_interaction().is_surface_interaction();
    }

    SampledSpectrum f(const Vertex &next, TransportMode mode) const {
        Vector3f wi = next.p() - p();

        if (wi.squared_length() == 0) {
            return 0;
        }

        wi = wi.normalize();
        switch (type) {
        case VertexType::surface: {
            return bsdf.f(si.wo, wi, mode);
        }
        case VertexType::medium: {
            return mi.medium->phase.eval(mi.wo, wi);
        }
        }

        REPORT_FATAL_ERROR();
        return NAN;
    }

    bool is_infinite_light() const {
        return type == VertexType::light && (!ei.light || ei.light->get_light_type() == LightType::infinite ||
                                             ei.light->get_light_type() == LightType::delta_direction);
    }

    Real convert_density(Real pdf, const Vertex &next) const {
        // Return solid angle density if _next_ is an infinite area light
        if (next.is_infinite_light()) {
            return pdf;
        }

        Vector3f w = next.p() - p();
        if (w.squared_length() == 0) {
            return 0;
        }

        Real invDist2 = 1 / w.squared_length();
        if (next.is_on_surface()) {
            pdf *= next.ng().abs_dot(w * std::sqrt(invDist2));
        }

        return pdf * invDist2;
    }
};

class BDPTIntegrator {
  public:
    BDPTIntegrator(const IntegratorBase *_base, int _max_depth, bool _regularize,
                   Sampler::Type _sampler_type = Sampler::Type::independent);

    static std::unique_ptr<BDPTIntegrator> create(const std::string &sampler_type, const ParameterDictionary &parameters,
                                                 const IntegratorBase *integrator_base);

    void render(Film *film, int samples_per_pixel) const;

    int generate_camera_subpath(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler, int maxDepth,
                                std::vector<Vertex> &path) const;

    int generate_light_subpath(SampledWavelengths &lambda, Sampler *sampler, int maxDepth,
                               std::vector<Vertex> &path) const;

    SampledSpectrum connect_bdpt(SampledWavelengths &lambda, std::vector<Vertex> &light_vertices,
                                 std::vector<Vertex> &camera_vertices, int s, int t, Sampler *sampler,
                                 pbrt::optional<Point2f> *pRaster) const;

    const IntegratorBase *const base;
    const int max_depth;
    const bool regularize;

  private:
    SampledSpectrum Li(std::vector<FilmSample> &film_samples, const Ray &ray, SampledWavelengths &lambda,
                        Sampler *sampler) const;

    int random_walk(SampledWavelengths &lambda, const Ray &initial_ray, Sampler *sampler, SampledSpectrum beta,
                     Real pdf, int maxDepth, TransportMode mode, Vertex *path) const;

    Real compute_mis_weight(std::vector<Vertex> &light_vertices, std::vector<Vertex> &camera_vertices,
                            Vertex &sampled, int s, int t) const;

    SampledSpectrum compute_G(const Vertex &v0, const Vertex &v1, const SampledWavelengths &lambda) const;

    Real infinite_light_density(Vector3f w) const;

    Real pdf_light(const Vertex &vertex, const Vertex &v) const;
    Real pdf(const Vertex &vertex, const Vertex *prev, const Vertex &next) const;
    Real pdf_light_origin(const Vertex &vertex, const Vertex &v) const;
    SampledSpectrum Le(const Vertex &vertex, const Vertex &v, const SampledWavelengths &lambda) const;

    std::vector<const Light *> infinite_lights;
    const Sampler::Type sampler_type;
};
