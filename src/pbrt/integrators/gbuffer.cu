#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <pbrt/accelerator/hlbvh.h>
#include <pbrt/base/camera.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/material.h>
#include <pbrt/base/sampler.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/integrators/gbuffer.h>
#include <pbrt/samplers/stratified.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/spectrum_util/rgb_color_space.h>
#include <stdexcept>

namespace {

constexpr int gbuffer_albedo_ray_split_depth = 2;
// Two pure-specular layers followed by one terminal material-albedo layer.
constexpr int gbuffer_albedo_max_specular_layers = 2;
const int lambda_sample_count = 5;
const int local_sample_count = 5;

PBRT_CPU_GPU
bool is_pure_specular(const BxDFFlags flags) {
    return pbrt::is_specular(flags) && !pbrt::is_non_specular(flags);
}

// Null materials are transparent medium boundaries. G-buffers describe the
// first actual surface; AO likewise ignores boundaries, but treats glass as
// geometry. max_distance is measured in world-space scene units.
PBRT_CPU_GPU
pbrt::optional<ShapeIntersection> intersect_surface(const IntegratorBase *base, Ray ray,
                                                    const Real max_distance = Infinity) {
    const auto &world_from_render = base->camera->get_camera_base()->camera_transform.world_from_render;
    const Point3f origin = world_from_render(ray.o);
    const Real direction_length = world_from_render(ray.d).length();
    bool previous_self_intersection = false;
    for (int bounce = 0; bounce < IntegratorBase::MAX_VOLUME_BOUNCES; ++bounce) {
        const Real remaining = max_distance - origin.distance(world_from_render(ray.o));
        if (remaining <= 0) {
            return {};
        }
        auto hit = base->intersect(ray, remaining / direction_length);
        if (!hit || hit->interaction.material) {
            return hit;
        }

        auto &surface = hit->interaction;
        surface.n = Normal3f(ray.d.normalize());
        ray = surface.spawn_ray(ray.d);
        if (hit->t_hit * direction_length < base->epsilon_distance) {
            if (previous_self_intersection) {
                ray.o += (base->epsilon_distance / direction_length) * ray.d;
            }
            previous_self_intersection = true;
        } else {
            previous_self_intersection = false;
        }
    }
    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
SampledSpectrum estimate_albedo_rho(const BSDF &bsdf, const Vector3f &wo, Sampler *sampler) {
    Real uc[local_sample_count];
    Point2f u[local_sample_count];
    for (int i = 0; i < local_sample_count; ++i) {
        uc[i] = sampler->get_1d();
        u[i] = sampler->get_2d();
    }
    return bsdf.rho(wo, uc, u, local_sample_count);
}

PBRT_CPU_GPU
RGB albedo_spectrum_to_rgb(const SampledSpectrum &albedo, const SampledWavelengths &lambda,
                           const GlobalSpectra *spectra) {
    const auto *color_space = spectra->rgb_color_space;
    const auto spectrum = (albedo * color_space->illuminant->sample(lambda)).safe_div(lambda.pdf_as_sampled_spectrum());
    const RGB xyz((spectra->cie_xyz[0]->sample(lambda) * spectrum).average(),
                  (spectra->cie_xyz[1]->sample(lambda) * spectrum).average(),
                  (spectra->cie_xyz[2]->sample(lambda) * spectrum).average());
    return color_space->rgb_from_xyz * (xyz / CIE_Y_integral);
}

PBRT_CPU_GPU
SampledSpectrum evaluate_next_surface_albedo(const IntegratorBase *base, const Ray &ray, SampledWavelengths &lambda,
                                             const int samples_per_pixel, const int pure_specular_layer,
                                             Sampler *sampler);

PBRT_CPU_GPU
SampledSpectrum evaluate_specular_albedo_sample(const IntegratorBase *base, const SurfaceInteraction &surface,
                                                const BSDF &bsdf, SampledWavelengths &lambda, Sampler *sampler,
                                                const int samples_per_pixel, const int pure_specular_layer) {
    const Vector3f wo = surface.wo;
    const BxDFFlags flags = bsdf.flags();
    SampledSpectrum albedo(0);
    const bool split =
        pure_specular_layer <= gbuffer_albedo_ray_split_depth && pbrt::is_reflective_and_transmissive(flags);

    if (split) {
        // Split only the first pure-specular layer. A second pure-specular
        // layer is still sampled, but follows one branch by probability.
        const auto samples = bsdf.sample_dielectric_f(wo, sampler->get_2d(), TransportMode::Radiance);
        const pbrt::optional<BSDFSample> *branches[] = {&samples.reflection, &samples.transmission};
        const Real direction_pdfs[] = {samples.reflection_direction_pdf, samples.transmission_direction_pdf};
        for (int branch_index = 0; branch_index < 2; ++branch_index) {
            const auto *branch = branches[branch_index];
            const Real direction_pdf = direction_pdfs[branch_index];
            if (!*branch || !(direction_pdf > 0)) {
                continue;
            }

            auto branch_lambda = lambda;
            const SampledSpectrum child_albedo = evaluate_next_surface_albedo(
                base, surface.spawn_ray((*branch)->wi), branch_lambda, samples_per_pixel, pure_specular_layer, sampler);
            // A dispersive child may terminate wavelengths. Convert its result
            // to this first-hit sample's wavelength-PDF basis before adding it.
            const auto child_in_parent_basis =
                child_albedo * lambda.pdf_as_sampled_spectrum().safe_div(branch_lambda.pdf_as_sampled_spectrum());
            albedo += (*branch)->f * (*branch)->wi.abs_dot(surface.shading.n.to_vector3()) * child_in_parent_basis /
                      direction_pdf;
        }
        return albedo;
    }

    // Beyond the split depth or with only one lobe, follow one BSDF sample.
    const auto branch = bsdf.sample_f(wo, sampler->get_1d(), sampler->get_2d(), TransportMode::Radiance);
    if (!branch) {
        return SampledSpectrum(0);
    }

    const Real pdf = branch->pdf_is_proportional ? bsdf.pdf(wo, branch->wi) : branch->pdf;
    if (!(pdf > 0)) {
        return SampledSpectrum(0);
    }

    auto branch_lambda = lambda;
    const SampledSpectrum child_albedo = evaluate_next_surface_albedo(
        base, surface.spawn_ray(branch->wi), branch_lambda, samples_per_pixel, pure_specular_layer, sampler);
    const auto child_in_parent_basis =
        child_albedo * lambda.pdf_as_sampled_spectrum().safe_div(branch_lambda.pdf_as_sampled_spectrum());
    return branch->f * branch->wi.abs_dot(surface.shading.n.to_vector3()) * child_in_parent_basis / pdf;
}

PBRT_CPU_GPU
SampledSpectrum evaluate_next_surface_albedo(const IntegratorBase *base, const Ray &ray, SampledWavelengths &lambda,
                                             const int samples_per_pixel, const int pure_specular_layer,
                                             Sampler *sampler) {
    auto hit = intersect_surface(base, ray);
    if (!hit) {
        // An escaped branch has no surface albedo contribution.
        return SampledSpectrum(0);
    }

    auto &surface = hit->interaction;
    auto hit_lambda = lambda;
    const auto bsdf = surface.get_bsdf(hit_lambda, base->camera, samples_per_pixel);
    lambda = hit_lambda;
    if (pure_specular_layer < gbuffer_albedo_max_specular_layers && is_pure_specular(bsdf.flags())) {
        return evaluate_specular_albedo_sample(base, surface, bsdf, lambda, sampler, samples_per_pixel,
                                               pure_specular_layer + 1);
    }
    // The third layer is terminal for this guide: estimate the hit material's
    // albedo without expanding another specular path.
    return estimate_albedo_rho(bsdf, surface.wo, sampler);
}

struct SurfaceAttributeSum {
    Real weight = 0;
    Real absolute_weight = 0;
    Real ao = 0;
    Vector3f normal{0, 0, 0};
    Real depth = 0;

    PBRT_CPU_GPU
    void add(const GBufferPixel &sample, const Real w) {
        if (w == 0) {
            return;
        }
        weight += w;
        absolute_weight += std::abs(w);
        normal += w * sample.normal;
        ao += w * sample.ambient_occlusion;
        depth += w * sample.depth;
    }

    PBRT_CPU_GPU
    bool has_valid_weight() const {
        return std::abs(weight) > 32 * std::numeric_limits<Real>::epsilon() * absolute_weight;
    }

    PBRT_CPU_GPU
    GBufferPixel average(const Real coverage) const {
        GBufferPixel pixel;
        pixel.normal = normal / weight;
        pixel.ambient_occlusion = ao / weight;
        pixel.depth = depth / weight;
        pixel.coverage = coverage;
        return pixel;
    }
};

struct AlbedoSum {
    Real weight = 0;
    Real absolute_weight = 0;
    RGB albedo{0};

    PBRT_CPU_GPU
    void add(const RGB &sample, const Real w) {
        if (w == 0) {
            return;
        }
        weight += w;
        absolute_weight += std::abs(w);
        albedo += w * sample;
    }

    PBRT_CPU_GPU
    bool has_valid_weight() const {
        return std::abs(weight) > 32 * std::numeric_limits<Real>::epsilon() * absolute_weight;
    }

    PBRT_CPU_GPU
    RGB average() const {
        return albedo / weight;
    }
};

PBRT_CPU_GPU
GBufferPixel evaluate_surface_sample(const IntegratorBase *base, const Real max_distance, const Ray &ray,
                                     const SurfaceInteraction &surface, Sampler *sampler) {
    GBufferPixel sample;
    const auto &transform = base->camera->get_camera_base()->camera_transform;
    sample.coverage = 1;

    // Keep the signed, world-space shading normal. Only the visualization is
    // mapped into RGB; AO separately orients its hemisphere toward the camera.
    sample.normal = transform.world_from_render(surface.shading.n).to_vector3().normalize();
    sample.depth = transform.camera_from_render(surface.p()).z;
    const Vector3f ao_normal = sample.normal.face_forward(-transform.world_from_render(ray.d));
    const Vector3f local_wi = sample_cosine_hemisphere(sampler->get_2d());
    const Vector3f world_wi = Frame::from_z(ao_normal).from_local(local_wi);
    const Ray shadow_ray = surface.spawn_ray(transform.render_from_world(world_wi));
    sample.ambient_occlusion = intersect_surface(base, shadow_ray, max_distance) ? 0 : 1;
    return sample;
}

PBRT_CPU_GPU
RGB evaluate_albedo_sample(const IntegratorBase *base, const GlobalSpectra *spectra, const SurfaceInteraction &surface,
                           const int samples_per_pixel, Sampler *sampler) {
    RGB albedo_rgb(0);
    // Continue the camera sample's stream through all albedo estimates.
    for (int i = 0; i < lambda_sample_count; ++i) {
        auto lambda = SampledWavelengths::sample_visible(sampler->get_1d());
        auto surface_copy = surface;
        const auto bsdf = surface_copy.get_bsdf(lambda, base->camera, samples_per_pixel);
        SampledSpectrum albedo;
        // Each first-hit sample uses its own wavelengths and BSDF. Descendant
        // surfaces continue this sample's path without another local loop.
        if (gbuffer_albedo_ray_split_depth > 0 && is_pure_specular(bsdf.flags())) {
            albedo = evaluate_specular_albedo_sample(base, surface_copy, bsdf, lambda, sampler, samples_per_pixel, 1);
        } else {
            albedo = estimate_albedo_rho(bsdf, surface_copy.wo, sampler);
        }
        albedo_rgb += albedo_spectrum_to_rgb(albedo, lambda, spectra);
    }
    return albedo_rgb / lambda_sample_count;
}

// One thread owns every sample for this pixel in each pass. The first pass
// initializes all fields; the second only updates albedo.
__global__ void render_gbuffer_non_albedo(const IntegratorBase *base, GBuffer *gbuffer, const Real max_distance,
                                          const int samples_per_pixel, const int num_pixels) {
    const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_index >= num_pixels) {
        return;
    }

    const auto resolution = gbuffer->get_resolution();
    const Point2i pixel(pixel_index % resolution.x, pixel_index / resolution.x);
    // Each thread owns its stratified sampler, with no full-image
    // initialization pass. render() has already rounded the budget to a square.
    StratifiedSampler stratified_sampler(samples_per_pixel);
    Sampler sampler(&stratified_sampler);
    SurfaceAttributeSum weighted, unweighted;
    int hits = 0;
    for (int i = 0; i < samples_per_pixel; ++i) {
        sampler.start_pixel_sample(pixel_index, i, 0);
        const auto camera_sample = sampler.get_camera_sample(pixel, base->filter);
        // Preserve the wavelength dimension so lens and AO samples match the
        // albedo pass and the original combined pass. No spectrum is needed.
        sampler.get_1d();
        const auto camera_ray = base->camera->generate_ray(camera_sample, &sampler);
        auto hit = intersect_surface(base, camera_ray.ray);
        if (!hit) {
            continue;
        }

        const auto sample = evaluate_surface_sample(base, max_distance, camera_ray.ray, hit->interaction, &sampler);
        ++hits;
        weighted.add(sample, camera_sample.filter_weight);
        unweighted.add(sample, 1);
    }

    GBufferPixel result;
    if (hits > 0) {
        // Attributes are conditional means over surface hits. Coverage remains
        // the unsigned hit fraction; preserve signed-filter reconstruction and
        // fall back to an unweighted mean only when its denominator cancels.
        result = (weighted.has_valid_weight() ? weighted : unweighted).average(Real(hits) / samples_per_pixel);
    }
    gbuffer->data()[pixel_index] = result;
}

__global__ void render_gbuffer_albedo(const IntegratorBase *base, const GlobalSpectra *spectra, GBuffer *gbuffer,
                                      const int samples_per_pixel, const int num_pixels) {
    const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_index >= num_pixels) {
        return;
    }

    const auto resolution = gbuffer->get_resolution();
    const Point2i pixel(pixel_index % resolution.x, pixel_index / resolution.x);
    StratifiedSampler stratified_sampler(samples_per_pixel);
    Sampler sampler(&stratified_sampler);
    AlbedoSum weighted, unweighted;
    int hits = 0;
    for (int i = 0; i < samples_per_pixel; ++i) {
        sampler.start_pixel_sample(pixel_index, i, 0);
        const auto camera_sample = sampler.get_camera_sample(pixel, base->filter);
        // Match the wavelength dimension reserved by the non-albedo pass so
        // both passes generate identical camera rays. Sample albedo wavelengths
        // independently at the first hit.
        sampler.get_1d();
        const auto camera_ray = base->camera->generate_ray(camera_sample, &sampler);
        auto hit = intersect_surface(base, camera_ray.ray);
        if (!hit) {
            continue;
        }

        const auto albedo = evaluate_albedo_sample(base, spectra, hit->interaction, samples_per_pixel, &sampler);
        ++hits;
        weighted.add(albedo, camera_sample.filter_weight);
        unweighted.add(albedo, 1);
    }

    // The preceding pass owns AO, normal, depth, and coverage.
    gbuffer->data()[pixel_index].albedo =
        hits > 0 ? (weighted.has_valid_weight() ? weighted : unweighted).average() : RGB(0);
}

} // namespace

GBufferIntegrator::GBufferIntegrator(const GlobalSpectra *spectra, const IntegratorBase *base,
                                     GPUMemoryAllocator &allocator)
    : base(base), spectra(spectra), max_distance(0) {
    const auto render_bounds = base->bvh->bounds();
    // Accept flat bounds (e.g. a single plane), but skip an empty scene's bounds.
    if (render_bounds.p_min.x <= render_bounds.p_max.x && render_bounds.p_min.y <= render_bounds.p_max.y &&
        render_bounds.p_min.z <= render_bounds.p_max.z) {
        const auto &world_from_render = base->camera->get_camera_base()->camera_transform.world_from_render;
        const auto world_bounds = world_from_render(render_bounds);
        max_distance = 0.05 * world_bounds.diagonal().length();
        if (!std::isfinite(max_distance)) {
            throw std::invalid_argument("GBufferIntegrator requires finite scene bounds");
        }
    }
    gbuffer = allocator.create<GBuffer>(base->camera->get_camera_base()->resolution, allocator);
}

void GBufferIntegrator::render(int samples_per_pixel) const {
    if (samples_per_pixel <= 0) {
        throw std::invalid_argument("GBufferIntegrator requires positive samples per pixel");
    }
    samples_per_pixel = StratifiedSampler::round_down_samples_per_pixel(samples_per_pixel);
    const auto resolution = gbuffer->get_resolution();
    const int num_pixels = resolution.x * resolution.y;
    printf("G-buffer sampling: %d spp (stratified)\n", samples_per_pixel);
    constexpr int threads = 128;
    render_gbuffer_non_albedo<<<divide_and_ceil(num_pixels, threads), threads>>>(base, gbuffer, max_distance,
                                                                                 samples_per_pixel, num_pixels);
    CHECK_CUDA_ERROR(cudaGetLastError());
    render_gbuffer_albedo<<<divide_and_ceil(num_pixels, threads), threads>>>(base, spectra, gbuffer, samples_per_pixel,
                                                                             num_pixels);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void GBufferIntegrator::write_images(const std::string &output_filename) const {
    const std::filesystem::path path(output_filename);
    const auto prefix = path.parent_path() / path.stem();
    // Write raw floating-point attributes before producing display mappings.
    const auto raw_filename = prefix.string() + "-gbuffer.exr";
    gbuffer->write_to_exr(raw_filename);
    const std::array<std::pair<const char *, GBufferChannel>, 4> outputs = {{
        {"ao", GBufferChannel::AmbientOcclusion},
        {"normal", GBufferChannel::SurfaceNormal},
        {"albedo", GBufferChannel::Albedo},
        {"depth", GBufferChannel::Depth},
    }};
    for (const auto &[suffix, output_channel] : outputs) {
        gbuffer->write_to_png(prefix.string() + "-" + suffix + ".png", output_channel);
    }
    printf("raw G-buffer saved to `%s`; AO, normal, albedo, depth visualizations saved alongside\n",
           raw_filename.c_str());
}
