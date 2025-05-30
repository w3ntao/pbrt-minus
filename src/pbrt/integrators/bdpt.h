#pragma once

#include <pbrt/base/light.h>

struct IntegratorBase;
struct FilmSample;

struct BDPTConfig {
    const IntegratorBase *base;

    uint max_depth;
    uint film_sample_size;

    bool regularize;
};

struct EndpointInteraction : Interaction {
    const Camera *camera;
    const Light *light;

    PBRT_CPU_GPU
    EndpointInteraction() : Interaction(), camera(nullptr), light(nullptr) {}

    PBRT_CPU_GPU
    EndpointInteraction(const Light *light, const Ray &r)
        : Interaction(r.o), camera(nullptr), light(light) {}

    PBRT_CPU_GPU
    EndpointInteraction(const Camera *camera, const Ray &ray)
        : Interaction(ray.o), camera(camera), light(nullptr) {}

    PBRT_CPU_GPU
    EndpointInteraction(const Light *light, const Interaction &intr)
        : Interaction(intr), camera(nullptr), light(light) {}

    PBRT_CPU_GPU
    EndpointInteraction(const Interaction &it, const Camera *camera)
        : Interaction(it), camera(camera), light(nullptr) {}

    PBRT_CPU_GPU
    EndpointInteraction(const Ray &ray)
        : Interaction(ray.at(1), Normal3f(-ray.d)), camera(nullptr), light(nullptr) {}
};

struct Vertex {
    enum class VertexType { camera, light, surface };

    VertexType type;
    SampledSpectrum beta;
    EndpointInteraction ei;
    SurfaceInteraction si;
    BSDF bsdf;

    bool delta;
    Real pdfFwd;
    Real pdfRev;

    PBRT_CPU_GPU
    Vertex() : type(VertexType::camera), beta(NAN), delta(false), pdfFwd(0), pdfRev(0) {}

    PBRT_CPU_GPU
    Vertex(VertexType _type, const EndpointInteraction &_ei, const SampledSpectrum &_beta)
        : type(_type), beta(_beta), delta(false), pdfFwd(0), pdfRev(0), ei(_ei) {}

    PBRT_CPU_GPU
    Vertex(const SurfaceInteraction &_si, const BSDF &_bsdf, const SampledSpectrum &_beta)
        : type(VertexType::surface), beta(_beta), delta(false), pdfFwd(0), pdfRev(0), si(_si),
          bsdf(_bsdf) {}

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
                               Real pdf) {
        Vertex v(VertexType::light, ei, beta);
        v.pdfFwd = pdf;
        return v;
    }

    PBRT_CPU_GPU
    static Vertex create_light(const Light *light, const Interaction &intr,
                               const SampledSpectrum &Le, Real pdf) {
        Vertex v(VertexType::light, EndpointInteraction(light, intr), Le);
        v.pdfFwd = pdf;
        return v;
    }

    PBRT_CPU_GPU
    static Vertex create_light(const Light *light, const Ray &ray, const SampledSpectrum &Le,
                               Real pdf) {
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
    bool is_connectible() const {
        switch (type) {
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
        SurfaceInteraction unused;
        return unused;
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
            return SampledSpectrum(0);
        }

        wi = wi.normalize();
        switch (type) {
        case VertexType::surface:
            return bsdf.f(si.wo, wi, mode);
        }

        REPORT_FATAL_ERROR();
        return SampledSpectrum(NAN);
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
    static BDPTIntegrator *create(int samples_per_pixel, const std::string &sampler_type,
                                  const ParameterDictionary &parameters,
                                  const IntegratorBase *integrator_base,
                                  GPUMemoryAllocator &allocator);

    void render(Film *film, uint samples_per_pixel, bool preview);

    PBRT_CPU_GPU
    static int generate_camera_subpath(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler,
                                       int maxDepth, Vertex *path, const BDPTConfig *config);

    PBRT_CPU_GPU
    static int generate_light_subpath(SampledWavelengths &lambda, Sampler *sampler, int maxDepth,
                                      Vertex *path, const BDPTConfig *config);

    PBRT_GPU
    static SampledSpectrum connect_bdpt(SampledWavelengths &lambda, Vertex *lightVertices,
                                        Vertex *cameraVertices, int s, int t, Sampler *sampler,
                                        pbrt::optional<Point2f> *pRaster, const BDPTConfig *config,
                                        Real *misWeightPtr = nullptr);

    PBRT_GPU
    static SampledSpectrum li(FilmSample *film_samples, int *film_sample_counter, const Ray &ray,
                              SampledWavelengths &lambda, Sampler *sampler, Vertex *camera_vertices,
                              Vertex *light_vertices, const BDPTConfig *config);

    const BDPTConfig *config;
    Sampler *samplers;
};
