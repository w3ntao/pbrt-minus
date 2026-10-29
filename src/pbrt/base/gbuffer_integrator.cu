#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <pbrt/accelerator/hlbvh.h>
#include <pbrt/base/camera.h>
#include <pbrt/base/gbuffer_integrator.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/sampler.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/spectrum_util/rgb_color_space.h>
#include <stdexcept>

namespace {

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

struct AttributeSum {
    Real weight = 0;
    Real absolute_weight = 0;
    Real ao = 0;
    Vector3f normal{0, 0, 0};
    RGB albedo{0};
    Real depth = 0;

    PBRT_CPU_GPU
    void add(const GBufferPixel &sample, const Real w) {
        if (w == 0) {
            return;
        }
        weight += w;
        absolute_weight += std::abs(w);
        ao += w * sample.ambient_occlusion;
        normal += w * sample.normal;
        albedo += w * sample.albedo;
        depth += w * sample.depth;
    }

    PBRT_CPU_GPU
    GBufferPixel average(const Real coverage) const {
        GBufferPixel pixel;
        pixel.ambient_occlusion = ao / weight;
        pixel.normal = normal / weight;
        pixel.albedo = albedo / weight;
        pixel.depth = depth / weight;
        pixel.coverage = coverage;
        return pixel;
    }
};

__global__ void render_gbuffer(const GBufferIntegrator *integrator, Sampler *samplers, const int samples_per_pixel,
                               const int num_pixels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_pixels) {
        integrator->render_pixel(index, samplers + index, samples_per_pixel);
    }
}

} // namespace

GBufferIntegrator::GBufferIntegrator(const ParameterDictionary &parameters, const IntegratorBase *base,
                                     GPUMemoryAllocator &allocator)
    : base(base), spectra(parameters.global_spectra), max_distance(0) {
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
    printf("G-buffer AO maxdistance: %.6g (2%% of world-space scene bounds diagonal)\n", double(max_distance));
    gbuffer = allocator.create<GBuffer>(base->camera->get_camera_base()->resolution, allocator);
}

PBRT_CPU_GPU
GBufferPixel GBufferIntegrator::evaluate_sample(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler,
                                                const int samples_per_pixel) const {
    GBufferPixel sample;
    auto hit = intersect_surface(base, ray);
    if (!hit) {
        return sample;
    }

    auto &surface = hit->interaction;
    const auto &transform = base->camera->get_camera_base()->camera_transform;
    const auto bsdf = surface.get_bsdf(lambda, base->camera, samples_per_pixel);
    // Keep the signed, world-space shading normal. Only the visualization is
    // mapped into RGB; AO separately orients its hemisphere toward the camera.
    sample.normal = transform.world_from_render(surface.shading.n).to_vector3().normalize();
    sample.depth = transform.camera_from_render(surface.p()).z;
    sample.coverage = 1;

    const Vector3f ao_normal = sample.normal.face_forward(-transform.world_from_render(ray.d));
    const Vector3f local_wi = sample_cosine_hemisphere(sampler->get_2d());
    const Vector3f world_wi = Frame::from_z(ao_normal).from_local(local_wi);
    const Ray shadow_ray = surface.spawn_ray(transform.render_from_world(world_wi));
    sample.ambient_occlusion = intersect_surface(base, shadow_ray, max_distance) ? 0 : 1;

    // Match PBRT-v4's albedo estimate: average 16 fixed BSDF samples with
    // reflection and transmission enabled, in radiance mode.
    constexpr int n_rho_samples = 16;
    const Real uc_rho[n_rho_samples] = {
        0.75741637, 0.37870818, 0.7083487, 0.18935409, 0.9149363, 0.35417435,
        0.5990858, 0.09467703, 0.8578725, 0.45746812, 0.686759, 0.17708716,
        0.9674518, 0.2995429, 0.5083201, 0.047338516};
    const Point2f u_rho[n_rho_samples] = {
        Point2f(0.855985, 0.570367), Point2f(0.381823, 0.851844),
        Point2f(0.285328, 0.764262), Point2f(0.733380, 0.114073),
        Point2f(0.542663, 0.344465), Point2f(0.127274, 0.414848),
        Point2f(0.964700, 0.947162), Point2f(0.594089, 0.643463),
        Point2f(0.095109, 0.170369), Point2f(0.825444, 0.263359),
        Point2f(0.429467, 0.454469), Point2f(0.244460, 0.816459),
        Point2f(0.756135, 0.731258), Point2f(0.516165, 0.152852),
        Point2f(0.180888, 0.214174), Point2f(0.898579, 0.503897)};
    const auto albedo = bsdf.rho(surface.wo, uc_rho, u_rho, n_rho_samples);

    // Project under the color space's reference white, independent of scene
    // illumination and exposure. BSDF construction may terminate secondary
    // wavelengths, so use the resulting wavelength PDFs for this projection.
    const auto *color_space = spectra->rgb_color_space;
    const auto spectrum =
        (albedo * color_space->illuminant->sample(lambda)).safe_div(lambda.pdf_as_sampled_spectrum());
    const RGB xyz((spectra->cie_xyz[0]->sample(lambda) * spectrum).average() / CIE_Y_integral,
                  (spectra->cie_xyz[1]->sample(lambda) * spectrum).average() / CIE_Y_integral,
                  (spectra->cie_xyz[2]->sample(lambda) * spectrum).average() / CIE_Y_integral);
    sample.albedo = color_space->rgb_from_xyz * xyz;
    return sample;
}

PBRT_CPU_GPU
void GBufferIntegrator::render_pixel(const int pixel_index, Sampler *sampler, const int samples_per_pixel) const {
    const auto resolution = gbuffer->get_resolution();
    const Point2i pixel(pixel_index % resolution.x, pixel_index / resolution.x);
    AttributeSum weighted, unweighted;
    int hits = 0;
    for (int i = 0; i < samples_per_pixel; ++i) {
        sampler->start_pixel_sample(pixel_index, i, 0);
        const auto camera_sample = sampler->get_camera_sample(pixel, base->filter);
        auto lambda = SampledWavelengths::sample_visible(sampler->get_1d());
        const auto camera_ray = base->camera->generate_ray(camera_sample, sampler);
        const auto sample = evaluate_sample(camera_ray.ray, lambda, sampler, samples_per_pixel);
        if (sample.coverage == 0) {
            continue;
        }
        ++hits;
        weighted.add(sample, camera_sample.filter_weight);
        unweighted.add(sample, 1);
    }

    GBufferPixel result;
    if (hits > 0) {
        // Attributes are conditional means over surface hits; coverage records
        // the independent, unsigned hit fraction. Preserve signed filter
        // reconstruction, falling back only if its denominator cancels out.
        const bool valid_weight =
            std::abs(weighted.weight) > 32 * std::numeric_limits<Real>::epsilon() * weighted.absolute_weight;
        result = (valid_weight ? weighted : unweighted).average(Real(hits) / samples_per_pixel);
    }
    gbuffer->data()[pixel_index] = result;
}

void GBufferIntegrator::render(const std::string &sampler_type, const int samples_per_pixel) const {
    if (samples_per_pixel <= 0) {
        throw std::invalid_argument("GBufferIntegrator requires positive samples per pixel");
    }
    const auto resolution = gbuffer->get_resolution();
    const int num_pixels = resolution.x * resolution.y;
    GPUMemoryAllocator local_allocator;
    auto *samplers = Sampler::create_samplers(sampler_type, samples_per_pixel, num_pixels, local_allocator);

    constexpr int threads = 128;
    render_gbuffer<<<divide_and_ceil(num_pixels, threads), threads>>>(this, samplers, samples_per_pixel, num_pixels);
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
    gbuffer->write_to_png(output_filename, GBufferChannel::AmbientOcclusion);
    printf("raw G-buffer saved to `%s`; AO, normal, albedo, depth visualizations saved alongside\n",
           raw_filename.c_str());
}
