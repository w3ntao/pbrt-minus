#pragma once

#include <pbrt/base/light.h>

struct IntegratorBase;
struct FilmSample;

struct BDPTConfig {
    const IntegratorBase *base = nullptr;

    int max_depth = 0;
    int film_sample_size = 0;

    bool regularize = true;

    BDPTConfig(const IntegratorBase *_base, const int _max_depth, const int _film_sample_size,
               const bool _regularize)
        : base(_base), max_depth(_max_depth), film_sample_size(_film_sample_size),
          regularize(_regularize) {}
};

struct EndpointInteraction : Interaction {
    const Camera *camera = nullptr;
    const Light *light = nullptr;

    PBRT_CPU_GPU
    EndpointInteraction() {}

    PBRT_CPU_GPU
    EndpointInteraction(const Light *_light, const Ray &ray)
        : Interaction(ray.o, ray.medium), light(_light) {}

    PBRT_CPU_GPU
    EndpointInteraction(const Camera *_camera, const Ray &ray)
        : Interaction(ray.o, ray.medium), camera(_camera) {}

    PBRT_CPU_GPU
    EndpointInteraction(const Light *_light, const Interaction &_interaction)
        : Interaction(_interaction), light(_light) {}

    PBRT_CPU_GPU
    EndpointInteraction(const Interaction &_interaction, const Camera *_camera)
        : Interaction(_interaction), camera(_camera) {}

    PBRT_CPU_GPU
    explicit EndpointInteraction(const Ray &ray)
        : Interaction(ray.at(1), Normal3f(-ray.d), ray.medium) {}
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

    PBRT_CPU_GPU
    Vertex() : type(VertexType::camera), beta(NAN) {}

    PBRT_CPU_GPU
    Vertex(const VertexType _type, const EndpointInteraction &_ei, const SampledSpectrum &_beta)
        : type(_type), ei(_ei), beta(_beta) {}

    PBRT_CPU_GPU
    Vertex(const SurfaceInteraction &_si, const BSDF &_bsdf, const SampledSpectrum &_beta)
        : type(VertexType::surface), si(_si), bsdf(_bsdf), beta(_beta) {}

    PBRT_CPU_GPU
    Vertex(const Interaction &mi, const SampledSpectrum &beta)
        : type(VertexType::medium), mi(mi), beta(beta) {}

    PBRT_CPU_GPU
    bool is_light() const {
        return type == VertexType::light ||
               (type == VertexType::surface && si.area_light != nullptr);
    }

    PBRT_CPU_GPU
    bool is_delta_light() const {
        return type == VertexType::light && ei.light &&
               pbrt::is_delta_light(ei.light->get_light_type());
    }

    PBRT_CPU_GPU
    static Vertex create_camera(const Camera *camera, const Ray &ray, const SampledSpectrum &beta) {
        return Vertex(VertexType::camera, EndpointInteraction(camera, ray), beta);
    }

    PBRT_CPU_GPU
    static Vertex create_camera(const Camera *camera, const Interaction &it,
                                const SampledSpectrum &beta) {
        return Vertex(VertexType::camera, EndpointInteraction(it, camera), beta);
    }

    PBRT_CPU_GPU
    static Vertex create_light(const EndpointInteraction &ei, const SampledSpectrum &beta,
                               const Real pdf) {
        Vertex v(VertexType::light, ei, beta);
        v.pdfFwd = pdf;
        return v;
    }

    PBRT_CPU_GPU
    static Vertex create_light(const Light *light, const Interaction &intractopm,
                               const SampledSpectrum &Le, const Real pdf) {
        Vertex v(VertexType::light, EndpointInteraction(light, intractopm), Le);
        v.pdfFwd = pdf;
        return v;
    }

    PBRT_CPU_GPU
    static Vertex create_light(const Light *light, const Ray &ray, const SampledSpectrum &Le,
                               const Real pdf) {
        Vertex v(VertexType::light, EndpointInteraction(light, ray), Le);
        v.pdfFwd = pdf;
        return v;
    }

    PBRT_CPU_GPU
    static Vertex create_surface(const SurfaceInteraction &si, const BSDF &bsdf,
                                 const SampledSpectrum &beta, Real pdf, const Vertex &prev) {
        Vertex v(si, bsdf, beta);
        v.pdfFwd = prev.convert_density(pdf, v);
        return v;
    }

    PBRT_CPU_GPU
    static Vertex create_medium(const Interaction &mi, const SampledSpectrum &beta, const Real pdf,
                                const Vertex &prev) {
        Vertex v(mi, beta);
        v.pdfFwd = prev.convert_density(pdf, v);
        return v;
    }

    PBRT_CPU_GPU
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

    PBRT_CPU_GPU
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

    PBRT_CPU_GPU
    const SurfaceInteraction &get_surface_interaction() const {
        if (type == VertexType::surface) {
            return si;
        }

        REPORT_FATAL_ERROR();
    }

    PBRT_CPU_GPU
    Point3f p() const {
        return get_interaction().p();
    }

    PBRT_CPU_GPU
    const Normal3f &ng() const {
        return get_interaction().n;
    }

    PBRT_CPU_GPU
    const Normal3f &ns() const {
        if (type == VertexType::surface) {
            return si.shading.n;
        }

        return get_interaction().n;
    }

    PBRT_CPU_GPU
    bool is_on_surface() const {
        return get_interaction().is_surface_interaction();
    }

    PBRT_CPU_GPU
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

    PBRT_CPU_GPU
    bool is_infinite_light() const {
        return type == VertexType::light &&
               (!ei.light || ei.light->get_light_type() == LightType::infinite ||
                ei.light->get_light_type() == LightType::delta_direction);
    }

    PBRT_CPU_GPU
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

    PBRT_CPU_GPU
    Real pdf_light(const IntegratorBase *integrator_base, const Vertex &v) const;

    PBRT_CPU_GPU
    Real pdf(const IntegratorBase *integrator_base, const Vertex *prev, const Vertex &next) const;

    PBRT_CPU_GPU
    SampledSpectrum Le(const Light **infinite_lights, int num_infinite_lights, const Vertex &v,
                       const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Real pdf_light_origin(const Light **infinite_lights, int num_infinite_lights, const Vertex &v,
                          const PowerLightSampler *lightSampler);
};

class BDPTIntegrator {
  public:
    BDPTIntegrator(const BDPTConfig *_config, Sampler *_samplers)
        : config(_config), samplers(_samplers) {}

    static BDPTIntegrator *create(int samples_per_pixel, const std::string &sampler_type,
                                  const ParameterDictionary &parameters,
                                  const IntegratorBase *integrator_base,
                                  GPUMemoryAllocator &allocator);

    void render(Film *film, int samples_per_pixel, bool preview);

    PBRT_CPU_GPU
    static int generate_camera_subpath(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler,
                                       int maxDepth, Vertex *path, const BDPTConfig *config);

    PBRT_CPU_GPU
    static int generate_light_subpath(SampledWavelengths &lambda, Sampler *sampler, int maxDepth,
                                      Vertex *path, const BDPTConfig *config);

    PBRT_GPU
    static SampledSpectrum connect_bdpt(SampledWavelengths &lambda, Vertex *lightVertices,
                                        Vertex *cameraVertices, int s, int t, Sampler *sampler,
                                        pbrt::optional<Point2f> *pRaster, const BDPTConfig *config);

    PBRT_GPU
    static SampledSpectrum Li(FilmSample *film_samples, int *film_sample_counter, const Ray &ray,
                              SampledWavelengths &lambda, Sampler *sampler, Vertex *camera_vertices,
                              Vertex *light_vertices, const BDPTConfig *config);

    const BDPTConfig *config = nullptr;
    Sampler *samplers = nullptr;
};
