#include "pbrt/base/interaction.h"

#include "pbrt/base/camera.h"
#include "pbrt/base/material.h"

#include "pbrt/bxdfs/diffuse_bxdf.h"

#include "pbrt/spectrum_util/sampled_wavelengths.h"

#include "pbrt/materials/diffuse_material.h"
// TODO: change DiffuseMaterial to Material
#include "pbrt/lights/diffuse_area_light.h"

PBRT_GPU
void SurfaceInteraction::compute_differentials(const Ray &ray, const Camera *camera,
                                               int samples_per_pixel) {
    // TODO: rewrite compute_differentials(): change Ray to RayDifferential
    const bool ray_has_differential = false;
    if (ray_has_differential) {
    } else {
        // Approximate screen-space change in $\pt{}$ based on camera projection
        // camera.Approximate_dp_dxy(p(), n, time, samplesPerPixel, &dpdx, &dpdy);
        camera->approximate_dp_dxy(pi.to_point3f(), n, samples_per_pixel, &dpdx, &dpdy);
    }

    // Estimate screen-space change in $(u,v)$
    // Compute $\transpose{\XFORM{A}} \XFORM{A}$ and its determinant
    FloatType ata00 = dpdu.dot(dpdu);
    FloatType ata01 = dpdu.dot(dpdv);
    FloatType ata11 = dpdv.dot(dpdv);

    FloatType invDet = 1.0 / difference_of_products(ata00, ata11, ata01, ata01);
    invDet = std::isfinite(invDet) ? invDet : 0.0;

    // Compute $\transpose{\XFORM{A}} \VEC{b}$ for $x$ and $y$
    FloatType atb0x = dpdu.dot(dpdx);
    FloatType atb1x = dpdv.dot(dpdx);
    FloatType atb0y = dpdu.dot(dpdy);
    FloatType atb1y = dpdv.dot(dpdy);

    // Compute $u$ and $v$ derivatives with respect to $x$ and $y$
    dudx = difference_of_products(ata11, atb0x, ata01, atb1x) * invDet;
    dvdx = difference_of_products(ata00, atb1x, ata01, atb0x) * invDet;
    dudy = difference_of_products(ata11, atb0y, ata01, atb1y) * invDet;
    dvdy = difference_of_products(ata00, atb1y, ata01, atb0y) * invDet;

    // Clamp derivatives of $u$ and $v$ to reasonable values
    dudx = std::isfinite(dudx) ? clamp(dudx, -1e8f, 1e8f) : 0.0;
    dvdx = std::isfinite(dvdx) ? clamp(dvdx, -1e8f, 1e8f) : 0.0;
    dudy = std::isfinite(dudy) ? clamp(dudy, -1e8f, 1e8f) : 0.0;
    dvdy = std::isfinite(dvdy) ? clamp(dvdy, -1e8f, 1e8f) : 0.0;
}

PBRT_GPU
void SurfaceInteraction::set_intersection_properties(const DiffuseMaterial *_material,
                                                     const DiffuseAreaLight *_area_light) {
    material = _material;
    area_light = _area_light;
}

PBRT_GPU
void SurfaceInteraction::init_diffuse_bsdf(BSDF &bsdf, DiffuseBxDF &diffuse_bxdf, const Ray &ray,
                                           SampledWavelengths &lambda, const Camera *camera,
                                           Sampler &sampler) {
    auto material_eval_context = MaterialEvalContext(*this);
    bsdf.init_frame(material_eval_context.ns, material_eval_context.dpdus);

    diffuse_bxdf = material->get_diffuse_bsdf(material_eval_context, lambda);
    bsdf.init_bxdf(&diffuse_bxdf);
}

PBRT_GPU
SampledSpectrum SurfaceInteraction::le(const Vector3f w, const SampledWavelengths &lambda) const {
    if (area_light == nullptr) {
        return SampledSpectrum::same_value(0.0);
    }

    return area_light->l(pi.to_point3f(), n, uv, w, lambda);
}
