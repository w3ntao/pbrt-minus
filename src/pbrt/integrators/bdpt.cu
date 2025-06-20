#include <pbrt/accelerator/hlbvh.h>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/sampler.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/gui/gl_helper.h>
#include <pbrt/integrators/bdpt.h>
#include <pbrt/light_samplers/power_light_sampler.h>
#include <pbrt/lights/image_infinite_light.h>
#include <pbrt/scene/parameter_dictionary.h>

constexpr size_t NUM_SAMPLERS = 64 * 1024;

struct FilmSample {
    Point2f p_film;
    SampledSpectrum l_path;
    SampledWavelengths lambda;

    // to help sorting
    PBRT_CPU_GPU
    bool operator<(const FilmSample &right) const {
        if (p_film.x < right.p_film.x) {
            return true;
        }
        if (p_film.x > right.p_film.x) {
            return false;
        }

        if (p_film.y < right.p_film.y) {
            return true;
        }
        if (p_film.y > right.p_film.y) {
            return false;
        }

        for (int idx = 0; idx < NSpectrumSamples; ++idx) {
            if (l_path[idx] < right.l_path[idx]) {
                return true;
            }
            if (l_path[idx] > right.l_path[idx]) {
                return false;
            }

            if (lambda[idx] < right.lambda[idx]) {
                return true;
            }
            if (lambda[idx] > right.lambda[idx]) {
                return false;
            }
        }

        return false;
    }
};

struct BDPTSample {
    Point2i p_pixel;
    Real weight;
    SampledSpectrum l_path;
    SampledWavelengths lambda;
};

PBRT_CPU_GPU
Real infinite_light_density(const Light **infinite_lights, int num_infinite_lights,
                            const PowerLightSampler *lightSampler, const Vector3f w) {
    Real pdf = 0;
    for (auto idx = 0; idx < num_infinite_lights; ++idx) {
        auto light = infinite_lights[idx];
        pdf += light->pdf_li(LightSampleContext(Interaction()), -w) * lightSampler->pmf(light);
    }

    return pdf;
}

PBRT_CPU_GPU
Real Vertex::pdf_light(const IntegratorBase *integrator_base, const Vertex &v) const {
    Vector3f w = v.p() - p();
    auto invDist2 = 1.0 / w.squared_length();
    w *= std::sqrt(invDist2);

    // Compute sampling density _pdf_ for light type
    Real pdf;
    if (is_infinite_light()) {
        // Compute planar sampling density for infinite light sources
        Bounds3f sceneBounds = integrator_base->bvh->bounds();
        Point3f sceneCenter;
        Real sceneRadius;
        sceneBounds.bounding_sphere(&sceneCenter, &sceneRadius);
        pdf = 1.0 / (compute_pi() * sqr(sceneRadius));
    } else if (is_on_surface()) {
        // Compute sampling density at emissive surface
        if (DEBUG_MODE && type == VertexType::light) {
            if (ei.light->get_light_type() != LightType::area) {
                REPORT_FATAL_ERROR();
            }
        }

        auto light = (type == VertexType::light) ? ei.light : si.area_light;
        Real pdfPos, pdfDir;
        light->pdf_le(ei, w, &pdfPos, &pdfDir);
        pdf = pdfDir * invDist2;
    } else {
        if (DEBUG_MODE) {
            if (type != VertexType::light || ei.light == nullptr) {
                REPORT_FATAL_ERROR();
            }
        }

        // Compute sampling density for noninfinite light sources
        Real pdfPos, pdfDir;
        ei.light->pdf_le(Ray(p(), w), &pdfPos, &pdfDir);
        pdf = pdfDir * invDist2;
    }

    if (v.is_on_surface()) {
        pdf *= v.ng().abs_dot(w);
    }

    return pdf;
}

PBRT_CPU_GPU
Real Vertex::pdf(const IntegratorBase *integrator_base, const Vertex *prev,
                 const Vertex &next) const {
    if (type == VertexType::light) {
        return pdf_light(integrator_base, next);
    }

    // Compute directions to preceding and next vertex
    Vector3f wn = next.p() - p();
    if (wn.squared_length() == 0) {
        return 0;
    }

    wn = wn.normalize();
    Vector3f wp;
    if (prev) {
        wp = prev->p() - p();
        if (wp.squared_length() == 0) {
            return 0;
        }
        wp = wp.normalize();
    } else {
        if (DEBUG_MODE && type != VertexType::camera) {
            REPORT_FATAL_ERROR();
        }
    }

    // Compute directional density depending on the vertex type
    Real pdf = 0;

    switch (type) {
    case VertexType::camera: {
        Real unused;
        ei.camera->pdf_we(ei.spawn_ray(wn), &unused, &pdf);
        break;
    }
    case VertexType::surface: {
        pdf = bsdf.pdf(wp, wn);
        break;
    }
    default: {
        REPORT_FATAL_ERROR();
    }
    }

    // Return probability per unit area at vertex _next_
    return convert_density(pdf, next);
}

PBRT_CPU_GPU
SampledSpectrum Vertex::Le(const Light **infinite_lights, int num_infinite_lights, const Vertex &v,
                           const SampledWavelengths &lambda) const {
    if (!is_light()) {
        return SampledSpectrum(0.0);
    }

    Vector3f w = v.p() - p();
    if (w.squared_length() == 0) {
        return SampledSpectrum(0.0);
    }

    w = w.normalize();
    if (is_infinite_light()) {
        // Return emitted radiance for infinite light sources
        SampledSpectrum Le(0.f);

        for (int idx = 0; idx < num_infinite_lights; ++idx) {
            auto light = infinite_lights[idx];
            Le += light->le(Ray(p(), -w), lambda);
        }

        return Le;
    }

    if (si.area_light != nullptr) {
        return si.area_light->l(si.p(), si.n, si.uv, w, lambda);
    }

    return SampledSpectrum(0.f);
}

PBRT_CPU_GPU
Real Vertex::pdf_light_origin(const Light **infinite_lights, int num_infinite_lights,
                              const Vertex &v, const PowerLightSampler *lightSampler) {
    Vector3f w = v.p() - p();
    if (w.squared_length() == 0) {
        return 0.0;
    }

    w = w.normalize();

    if (is_infinite_light()) {
        // Return sampling density for infinite light sources
        return infinite_light_density(infinite_lights, num_infinite_lights, lightSampler, w);
    }

    // Return sampling density for noninfinite light source
    auto light = type == VertexType::light ? ei.light : si.area_light;

    Real pdfPos, pdfDir;
    auto pdfChoice = lightSampler->pmf(light);

    if (is_on_surface()) {
        light->pdf_le(ei, w, &pdfPos, &pdfDir);
    } else {
        light->pdf_le(Ray(p(), w), &pdfPos, &pdfDir);
    }

    return pdfPos * pdfChoice;
}

template <typename Type>
class ScopedAssignment {
  public:
    PBRT_CPU_GPU
    explicit ScopedAssignment(Type *_target = nullptr, Type value = Type()) : target(_target) {
        if (_target) {
            backup = *_target;
            *_target = value;
        }
    }

    PBRT_CPU_GPU
    void assign() {
        if (target)
            *target = backup;
    }

    PBRT_CPU_GPU
    ScopedAssignment &operator=(ScopedAssignment &&other) {
        target = other.target;
        backup = other.backup;
        other.target = nullptr;
        return *this;
    }

  private:
    Type *target;
    Type backup;
};

PBRT_CPU_GPU
SampledSpectrum G(const IntegratorBase *integrator_base, const Vertex &v0, const Vertex &v1,
                  const SampledWavelengths &lambda) {
    Vector3f d = v0.p() - v1.p();
    auto g = 1.0 / d.squared_length();
    d *= std::sqrt(g);
    if (v0.is_on_surface()) {
        g *= v0.ns().abs_dot(d);
    }

    if (v1.is_on_surface()) {
        g *= v1.ns().abs_dot(d);
    }

    return g * integrator_base->tr(v0.get_interaction(), v1.get_interaction());
}

PBRT_CPU_GPU
Real mis_weight(const IntegratorBase *integrator_base, Vertex *lightVertices,
                Vertex *cameraVertices, Vertex &sampled, int s, int t) {
    if (s + t == 2) {
        return 1;
    }

    // Define helper function _remap0_ that deals with Dirac delta functions
    auto remap0 = [](float f) -> Real { return f != 0 ? f : 1.0; };

    // Temporarily update vertex properties for current strategy
    // Look up connection vertices and their predecessors
    Vertex *qs = s > 0 ? &lightVertices[s - 1] : nullptr,
           *pt = t > 0 ? &cameraVertices[t - 1] : nullptr,
           *qsMinus = s > 1 ? &lightVertices[s - 2] : nullptr,
           *ptMinus = t > 1 ? &cameraVertices[t - 2] : nullptr;

    // Update sampled vertex for $s=1$ or $t=1$ strategy
    ScopedAssignment<Vertex> a1;
    if (s == 1) {
        a1 = ScopedAssignment(qs, sampled);
    } else if (t == 1) {
        a1 = ScopedAssignment(pt, sampled);
    }

    // Mark connection vertices as non-degenerate
    ScopedAssignment<bool> a2, a3;
    if (pt) {
        a2 = ScopedAssignment(&pt->delta, false);
    }
    if (qs) {
        a3 = ScopedAssignment(&qs->delta, false);
    }

    // Update reverse density of vertex $\pt{}_{t-1}$
    ScopedAssignment<Real> a4;
    if (pt) {
        a4 = ScopedAssignment(
            &pt->pdfRev, s > 0 ? qs->pdf(integrator_base, qsMinus, *pt)
                               : pt->pdf_light_origin(integrator_base->infinite_lights,
                                                      integrator_base->infinite_light_num, *ptMinus,
                                                      integrator_base->light_sampler));
    }

    // Update reverse density of vertex $\pt{}_{t-2}$
    ScopedAssignment<Real> a5;
    if (ptMinus) {
        a5 = ScopedAssignment(&ptMinus->pdfRev, s > 0 ? pt->pdf(integrator_base, qs, *ptMinus)
                                                      : pt->pdf_light(integrator_base, *ptMinus));
    }

    // Update reverse density of vertices $\pq{}_{s-1}$ and $\pq{}_{s-2}$
    ScopedAssignment<Real> a6;
    if (qs) {
        a6 = ScopedAssignment(&qs->pdfRev, pt->pdf(integrator_base, ptMinus, *qs));
    }

    ScopedAssignment<Real> a7;
    if (qsMinus) {
        a7 = ScopedAssignment(&qsMinus->pdfRev, qs->pdf(integrator_base, pt, *qsMinus));
    }

    Real sumRi = 0;

    // Consider hypothetical connection strategies along the camera subpath
    Real ri = 1.0;
    for (int i = t - 1; i > 0; --i) {
        ri *= remap0(cameraVertices[i].pdfRev) / remap0(cameraVertices[i].pdfFwd);
        if (!cameraVertices[i].delta && !cameraVertices[i - 1].delta) {
            sumRi += ri;
        }
    }

    ri = 1;
    for (int i = s - 1; i >= 0; --i) {
        ri *= remap0(lightVertices[i].pdfRev) / remap0(lightVertices[i].pdfFwd);
        bool deltaLightvertex =
            i > 0 ? lightVertices[i - 1].delta : lightVertices[0].is_delta_light();
        if (!lightVertices[i].delta && !deltaLightvertex) {
            sumRi += ri;
        }
    }

    a1.assign();
    a2.assign();
    a3.assign();
    a4.assign();
    a5.assign();
    a6.assign();
    a7.assign();

    return 1.0 / (1.0 + sumRi);
}

PBRT_CPU_GPU
int random_walk(SampledWavelengths &lambda, Ray ray, Sampler *sampler, SampledSpectrum beta,
                Real pdf, int maxDepth, TransportMode mode, Vertex *path,
                const BDPTConfig *config) {
    if (maxDepth == 0) {
        return 0;
    }

    const auto camera = config->base->camera;

    // Follow random walk to initialize BDPT path vertices
    int bounces = 0;
    bool anyNonSpecularBounces = false;
    auto pdfFwd = pdf;
    while (true) {
        if (!beta.is_positive()) {
            break;
        }

        bool scattered = false;
        bool terminated = false;

        // Trace a ray and sample the medium, if any
        Vertex &vertex = path[bounces];
        Vertex &prev = path[bounces - 1];
        auto si = config->base->intersect(ray, Infinity);

        if (terminated) {
            return bounces;
        }
        if (scattered) {
            continue;
        }

        // Handle escaped rays after no medium scattering event
        if (!si) {
            // Capture escaped rays when tracing from the camera
            if (mode == TransportMode::Radiance) {
                vertex = Vertex::create_light(EndpointInteraction(ray), beta, pdfFwd);
                ++bounces;
            }
            break;
        }

        // Handle surface interaction for path generation
        SurfaceInteraction &isect = si->interaction;
        // Get BSDF and skip over medium boundaries

        auto bsdf = isect.get_bsdf(lambda, camera, sampler->get_samples_per_pixel());

        // Possibly regularize the BSDF
        if (config->regularize && anyNonSpecularBounces) {
            bsdf.regularize();
        }

        // Initialize _vertex_ with surface intersection information
        vertex = Vertex::create_surface(isect, bsdf, beta, pdfFwd, prev);

        if (++bounces >= maxDepth) {
            break;
        }

        // Sample BSDF at current vertex
        Vector3f wo = isect.wo;
        auto u = sampler->get_1d();

        auto bs = vertex.bsdf.sample_f(wo, u, sampler->get_2d(), mode);
        if (!bs) {
            break;
        }

        pdfFwd = bs->pdf_is_proportional ? vertex.bsdf.pdf(wo, bs->wi, mode) : bs->pdf;
        anyNonSpecularBounces |= !bs->is_specular();

        beta *= bs->f * isect.shading.n.abs_dot(bs->wi) / bs->pdf;
        ray = isect.spawn_ray(bs->wi);
        // spawn_ray() is simplified from the original one from PBRT-v4

        auto _pdfRev = vertex.bsdf.pdf(bs->wi, wo, !mode);

        if (bs->is_specular()) {
            vertex.delta = true;
            _pdfRev = pdfFwd = 0;
        }

        prev.pdfRev = vertex.convert_density(_pdfRev, prev);
    }

    return bounces;
}

PBRT_CPU_GPU
int BDPTIntegrator::generate_camera_subpath(const Ray &ray, SampledWavelengths &lambda,
                                            Sampler *sampler, int maxDepth, Vertex *path,
                                            const BDPTConfig *config) {
    if (maxDepth == 0) {
        return 0;
    }

    const auto camera = config->base->camera;

    SampledSpectrum beta(1.f);
    // Generate first vertex on camera subpath and start random walk
    Real pdfPos, pdfDir;

    path[0] = Vertex::create_camera(camera, ray, beta);

    camera->pdf_we(ray, &pdfPos, &pdfDir);

    return random_walk(lambda, ray, sampler, beta, pdfDir, maxDepth - 1, TransportMode::Radiance,
                       path + 1, config) +
           1;
}

PBRT_CPU_GPU
int BDPTIntegrator::generate_light_subpath(SampledWavelengths &lambda, Sampler *sampler,
                                           int maxDepth, Vertex *path, const BDPTConfig *config) {
    // Generate light subpath and initialize _path_ vertices
    if (maxDepth == 0) {
        return 0;
    }

    const auto integrator_base = config->base;

    // Sample initial ray for light subpath
    // Sample light for BDPT light subpath
    auto sampledLight = integrator_base->light_sampler->sample(sampler->get_1d());
    if (!sampledLight) {
        return 0;
    }

    auto light = sampledLight->light;
    auto lightSamplePDF = sampledLight->p;

    auto ul0 = sampler->get_2d();
    auto ul1 = sampler->get_2d();
    auto les = light->sample_le(ul0, ul1, lambda);

    if (!les || les->pdfPos == 0 || les->pdfDir == 0 || !les->L.is_positive()) {
        return 0;
    }

    auto ray = les->ray;

    // Generate first vertex of light subpath
    auto p_l = lightSamplePDF * les->pdfPos;
    path[0] = les->intr ? Vertex::create_light(light, *les->intr, les->L, p_l)
                        : Vertex::create_light(light, ray, les->L, p_l);

    // Follow light subpath random walk
    SampledSpectrum beta = les->L * les->abs_cos_theta(ray.d) / (p_l * les->pdfDir);

    int nVertices = random_walk(lambda, ray, sampler, beta, les->pdfDir, maxDepth - 1,
                                TransportMode::Importance, path + 1, config);

    // Correct subpath sampling densities for infinite area lights
    if (path[0].is_infinite_light()) {
        // Set spatial density of _path[1]_ for infinite area light
        if (nVertices > 0) {
            path[1].pdfFwd = les->pdfPos;
            if (path[1].is_on_surface()) {
                path[1].pdfFwd *= path[1].ng().abs_dot(ray.d);
            }
        }

        // Set spatial density of _path[0]_ for infinite area light
        path[0].pdfFwd = infinite_light_density(integrator_base->infinite_lights,
                                                integrator_base->infinite_light_num,
                                                integrator_base->light_sampler, ray.d);
    }

    return nVertices + 1;
}

PBRT_GPU
SampledSpectrum BDPTIntegrator::connect_bdpt(SampledWavelengths &lambda, Vertex *lightVertices,
                                             Vertex *cameraVertices, int s, int t, Sampler *sampler,
                                             pbrt::optional<Point2f> *pRaster,
                                             const BDPTConfig *config, Real *misWeightPtr) {
    SampledSpectrum L(0.f);
    // Ignore invalid connections related to infinite area lights
    if (t > 1 && s != 0 && cameraVertices[t - 1].type == Vertex::VertexType::light) {
        return SampledSpectrum(0);
    }

    const auto integrator_base = config->base;

    const auto camera = integrator_base->camera;

    // Perform connection and write contribution to _L_
    Vertex sampled;
    if (s == 0) {
        // Interpret the camera subpath as a complete path
        const Vertex &pt = cameraVertices[t - 1];
        if (pt.is_light()) {
            L = pt.Le(integrator_base->infinite_lights, integrator_base->infinite_light_num,
                      cameraVertices[t - 2], lambda) *
                pt.beta;
        }
    } else if (t == 1) {
        // Sample a point on the camera and connect it to the light subpath
        const Vertex &qs = lightVertices[s - 1];
        if (qs.is_connectible()) {
            auto cs = camera->sample_wi(qs.get_interaction(), sampler->get_2d(), lambda);
            if (cs) {
                *pRaster = cs->pRaster;
                // Initialize dynamically sampled vertex and _L_ for $t=1$ case
                sampled = Vertex::create_camera(camera, cs->pLens, cs->Wi / cs->pdf);

                L = qs.beta * qs.f(sampled, TransportMode::Importance) * sampled.beta;
                if (qs.is_on_surface()) {
                    L *= qs.ns().abs_dot(cs->wi);
                }

                if (L.is_positive()) {
                    L *= integrator_base->tr(cs->pRef, cs->pLens);
                }
            }
        }
    } else if (s == 1) {
        // Sample a point on a light and connect it to the camera subpath
        const Vertex &pt = cameraVertices[t - 1];
        if (pt.is_connectible()) {
            auto sampledLight = integrator_base->light_sampler->sample(sampler->get_1d());
            if (sampledLight) {
                auto light = sampledLight->light;
                auto p_l = sampledLight->p;

                LightSampleContext ctx;
                if (pt.is_on_surface()) {
                    const SurfaceInteraction &si = pt.get_surface_interaction();
                    ctx = LightSampleContext(si);
                    // Try to nudge the light sampling position to correct side of the
                    // surface
                    BxDFFlags flags = pt.bsdf.flags();
                    if (pbrt::is_reflective(flags) && !pbrt::is_transmissive(flags)) {
                        ctx.pi = si.offset_ray_origin(si.wo);
                    } else if (pbrt::is_transmissive(flags) && !pbrt::is_reflective(flags)) {
                        ctx.pi = si.offset_ray_origin(-si.wo);
                    }
                } else {
                    ctx = LightSampleContext(pt.get_interaction());
                }

                auto lightWeight = light->sample_li(ctx, sampler->get_2d(), lambda);
                if (lightWeight && lightWeight->l.is_positive() && lightWeight->pdf > 0) {
                    EndpointInteraction ei(light, lightWeight->p_light);

                    sampled =
                        Vertex::create_light(ei, lightWeight->l / (lightWeight->pdf * p_l), 0);
                    sampled.pdfFwd = sampled.pdf_light_origin(integrator_base->infinite_lights,
                                                              integrator_base->infinite_light_num,
                                                              pt, integrator_base->light_sampler);

                    L = pt.beta * pt.f(sampled, TransportMode::Radiance) * sampled.beta;

                    if (pt.is_on_surface()) {
                        L *= pt.ns().abs_dot(lightWeight->wi);
                    }

                    // Only check visibility if the path would carry radiance.
                    if (L.is_positive()) {
                        L *= integrator_base->tr(pt.get_interaction(), lightWeight->p_light);
                    }
                }
            }
        }
    } else {
        // Handle all other bidirectional connection cases
        const Vertex &qs = lightVertices[s - 1], &pt = cameraVertices[t - 1];
        if (qs.is_connectible() && pt.is_connectible()) {
            L = qs.beta * qs.f(pt, TransportMode::Importance) * pt.f(qs, TransportMode::Radiance) *
                pt.beta;

            if (L.is_positive()) {
                L *= G(integrator_base, qs, pt, lambda);
            }
        }
    }

    // Compute MIS weight for connection strategy
    Real misWeight = L.is_positive()
                         ? mis_weight(integrator_base, lightVertices, cameraVertices, sampled, s, t)
                         : 0.0;

    L *= misWeight;
    if (misWeightPtr != nullptr) {
        *misWeightPtr = misWeight;
    }

    return L;
}

BDPTIntegrator *BDPTIntegrator::create(int samples_per_pixel, const std::string &sampler_type,
                                       const ParameterDictionary &parameters,
                                       const IntegratorBase *integrator_base,
                                       GPUMemoryAllocator &allocator) {
    auto bdpt_integrator = allocator.allocate<BDPTIntegrator>();
    auto config = allocator.allocate<BDPTConfig>();

    config->base = integrator_base;
    config->max_depth = parameters.get_integer("maxdepth", 5);
    config->regularize = parameters.get_bool("regularize", false);
    config->film_sample_size = config->max_depth * NUM_SAMPLERS;

    bdpt_integrator->config = config;

    bdpt_integrator->samplers =
        Sampler::create_samplers(sampler_type, samples_per_pixel, NUM_SAMPLERS, allocator);

    return bdpt_integrator;
}

__global__ void wavefront_render(BDPTSample *bdpt_samples, FilmSample *film_samples,
                                 int *film_sample_counter, Vertex *global_camera_vertices,
                                 Vertex *global_light_vertices, int pass, int samples_per_pixel,
                                 const Point2i film_resolution, BDPTIntegrator *bdpt_integrator) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= NUM_SAMPLERS) {
        return;
    }

    const auto width = film_resolution.x;
    const auto height = film_resolution.y;

    auto global_idx = (long long)(pass)*NUM_SAMPLERS + worker_idx;

    const auto pixel_idx = global_idx % (width * height);
    const auto sample_idx = global_idx / (width * height);
    if (sample_idx >= samples_per_pixel) {
        return;
    }

    const auto config = bdpt_integrator->config;

    auto local_sampler = &bdpt_integrator->samplers[worker_idx];
    local_sampler->start_pixel_sample(pixel_idx, sample_idx, 0);

    auto p_pixel = Point2i(pixel_idx % width, pixel_idx / width);

    auto camera_sample = local_sampler->get_camera_sample(p_pixel, config->base->filter);

    auto lu = local_sampler->get_1d();
    auto lambda = SampledWavelengths::sample_visible(lu);

    auto ray = config->base->camera->generate_ray(camera_sample, local_sampler);

    auto local_camera_vertices = &global_camera_vertices[worker_idx * (config->max_depth + 2)];
    auto local_light_vertices = &global_light_vertices[worker_idx * (config->max_depth + 1)];

    auto radiance_l = ray.weight * bdpt_integrator->li(film_samples, film_sample_counter, ray.ray,
                                                       lambda, local_sampler, local_camera_vertices,
                                                       local_light_vertices, config);

    auto rendered_sample = &bdpt_samples[worker_idx];
    rendered_sample->p_pixel = p_pixel;
    rendered_sample->weight = camera_sample.filter_weight;
    rendered_sample->l_path = radiance_l;
    rendered_sample->lambda = lambda;
}

void BDPTIntegrator::render(Film *film, int samples_per_pixel, const bool preview) {
    const auto image_resolution = film->get_resolution();

    GPUMemoryAllocator local_allocator;

    GLHelper gl_helper;
    if (preview) {
        gl_helper.init("initializing", image_resolution);
    }

    auto bdpt_samples = local_allocator.allocate<BDPTSample>(NUM_SAMPLERS);

    auto film_samples = local_allocator.allocate<FilmSample>(config->film_sample_size);

    auto film_sample_counter = local_allocator.allocate<int>();

    auto global_camera_vertices =
        local_allocator.allocate<Vertex>(NUM_SAMPLERS * (config->max_depth + 2));
    auto global_light_vertices =
        local_allocator.allocate<Vertex>(NUM_SAMPLERS * (config->max_depth + 1));

    auto num_pixels = image_resolution.x * image_resolution.y;

    constexpr int threads = 32;
    const int blocks = divide_and_ceil<int>(NUM_SAMPLERS, threads);

    auto total_pass = divide_and_ceil<long long>(num_pixels * samples_per_pixel, NUM_SAMPLERS);

    for (int pass = 0; pass < total_pass; ++pass) {
        *film_sample_counter = 0;
        wavefront_render<<<blocks, threads>>>(bdpt_samples, film_samples, film_sample_counter,
                                              global_camera_vertices, global_light_vertices, pass,
                                              samples_per_pixel, film->get_resolution(), this);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        for (int idx = 0; idx < NUM_SAMPLERS; ++idx) {
            const auto global_idx = (long long)(pass)*NUM_SAMPLERS + idx;
            const auto sample_idx = global_idx / num_pixels;
            if (sample_idx >= samples_per_pixel) {
                return;
            }

            const auto sample = &bdpt_samples[idx];
            film->add_sample(sample->p_pixel, sample->l_path, sample->lambda, sample->weight);
        }

        if (*film_sample_counter > 0) {
            // sort to make film writing deterministic
            std::sort(film_samples + 0, film_samples + (*film_sample_counter), std::less{});
        }

        for (int idx = 0; idx < *film_sample_counter; ++idx) {
            const auto sample = &film_samples[idx];
            film->add_splat(sample->p_film, sample->l_path, sample->lambda);
        }

        if (preview) {
            film->copy_to_frame_buffer(gl_helper.gpu_frame_buffer, 1.0 / samples_per_pixel);
            gl_helper.draw_frame(GLHelper::assemble_title(Real(pass + 1) / total_pass));
        }
    }
}

PBRT_GPU
SampledSpectrum BDPTIntegrator::li(FilmSample *film_samples, int *film_sample_counter,
                                   const Ray &ray, SampledWavelengths &lambda, Sampler *sampler,
                                   Vertex *camera_vertices, Vertex *light_vertices,
                                   const BDPTConfig *config) {
    // Trace the camera and light subpaths
    int nCamera = generate_camera_subpath(ray, lambda, sampler, config->max_depth + 2,
                                          camera_vertices, config);
    int nLight =
        generate_light_subpath(lambda, sampler, config->max_depth + 1, light_vertices, config);

    SampledSpectrum accumulated_l(0);

    // Execute all BDPT connection strategies
    for (int t = 1; t <= nCamera; ++t) {
        for (int s = 0; s <= nLight; ++s) {
            int depth = t + s - 2;
            if ((s == 1 && t == 1) || depth < 0 || depth > config->max_depth) {
                continue;
            }

            // Execute the $(s, t)$ connection strategy and update _L_
            pbrt::optional<Point2f> optional_p_film_new;
            Real misWeight = 0;
            SampledSpectrum l_path =
                connect_bdpt(lambda, light_vertices, camera_vertices, s, t, sampler,
                             &optional_p_film_new, config, &misWeight);

            if (t != 1) {
                accumulated_l += l_path;
            } else if (l_path.is_positive()) {
                if (DEBUG_MODE && !optional_p_film_new.has_value()) {
                    REPORT_FATAL_ERROR();
                }

                const auto film_sample_idx = atomicAdd(film_sample_counter, 1);

                if (film_sample_idx >= config->film_sample_size) {
                    REPORT_FATAL_ERROR();
                }

                film_samples[film_sample_idx].p_film = optional_p_film_new.value();
                film_samples[film_sample_idx].l_path = l_path;
                film_samples[film_sample_idx].lambda = lambda;
            }
        }
    }

    return accumulated_l;
}
