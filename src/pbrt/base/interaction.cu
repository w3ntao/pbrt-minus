#include "pbrt/base/camera.h"
#include "pbrt/base/interaction.h"
#include "pbrt/base/light.h"
#include "pbrt/base/material.h"
#include "pbrt/base/sampler.h"
#include "pbrt/bxdfs/coated_conductor_bxdf.h"
#include "pbrt/bxdfs/coated_diffuse_bxdf.h"
#include "pbrt/bxdfs/conductor_bxdf.h"
#include "pbrt/bxdfs/dielectric_bxdf.h"
#include "pbrt/bxdfs/diffuse_bxdf.h"

PBRT_GPU
void SurfaceInteraction::compute_differentials(const Ray &ray, const Camera *camera,
                                               uint samples_per_pixel) {
    // different with PBRT-v4: ignore the DifferentialRay
    camera->approximate_dp_dxy(p(), n, samples_per_pixel, &dpdx, &dpdy);

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
    dudx = std::isfinite(dudx) ? clamp<FloatType>(dudx, -1e8f, 1e8f) : 0.0;
    dvdx = std::isfinite(dvdx) ? clamp<FloatType>(dvdx, -1e8f, 1e8f) : 0.0;
    dudy = std::isfinite(dudy) ? clamp<FloatType>(dudy, -1e8f, 1e8f) : 0.0;
    dvdy = std::isfinite(dvdy) ? clamp<FloatType>(dvdy, -1e8f, 1e8f) : 0.0;
}

PBRT_GPU
void SurfaceInteraction::set_intersection_properties(const Material *_material,
                                                     const Light *_area_light) {
    if (_material->get_material_type() == Material::Type::mix) {
        material = _material->get_mix_material(this);
    } else {
        material = _material;
    }

    area_light = _area_light;
}

PBRT_CPU_GPU
void SurfaceInteraction::set_shading_geometry(const Normal3f &ns, const Vector3f &dpdus,
                                              const Vector3f &dpdvs, const Normal3f &dndus,
                                              const Normal3f &dndvs,
                                              bool orientationIsAuthoritative) {
    // Compute _shading.n_ for _SurfaceInteraction_
    shading.n = ns;

    if (orientationIsAuthoritative) {
        n = n.face_forward(shading.n);
    } else {
        shading.n = shading.n.face_forward(n);
    }

    // Initialize _shading_ partial derivative values
    shading.dpdu = dpdus;
    shading.dpdv = dpdvs;
    shading.dndu = dndus;
    shading.dndv = dndvs;

    while (shading.dpdu.squared_length() > 1e16f || shading.dpdv.squared_length() > 1e16f) {
        shading.dpdu /= 1e8f;
        shading.dpdv /= 1e8f;
    }
}

PBRT_GPU
void SurfaceInteraction::init_bsdf(BSDF &bsdf, FullBxDF &full_bxdf, const Ray &ray,
                                   SampledWavelengths &lambda, const Camera *camera,
                                   uint samples_per_pixel) {
    compute_differentials(ray, camera, samples_per_pixel);

    auto material_eval_context = MaterialEvalContext(*this);
    bsdf.init_frame(material_eval_context.ns, material_eval_context.dpdus);

    switch (material->get_material_type()) {
    case Material::Type::coated_conductor: {
        init_coated_conductor_bsdf(bsdf, full_bxdf.coated_conductor_bxdf, lambda,
                                   material_eval_context);
        break;
    }
    case Material::Type::coated_diffuse: {
        init_coated_diffuse_bsdf(bsdf, full_bxdf.coated_diffuse_bxdf, lambda,
                                 material_eval_context);
        break;
    }

    case Material::Type::conductor: {
        init_conductor_bsdf(bsdf, full_bxdf.conductor_bxdf, lambda, material_eval_context);
        break;
    }

    case Material::Type::dielectric: {
        init_dielectric_bsdf(bsdf, full_bxdf.dielectric_bxdf, lambda, material_eval_context);
        break;
    }

    case Material::Type::diffuse: {
        init_diffuse_bsdf(bsdf, full_bxdf.diffuse_bxdf, lambda, material_eval_context);
        break;
    }

    case Material::Type::mix: {
        printf("\nyou should not see MixMaterial here\n\n");
        REPORT_FATAL_ERROR();
    }

    default: {
        REPORT_FATAL_ERROR();
    }
    }
}

PBRT_GPU
void SurfaceInteraction::init_coated_conductor_bsdf(
    BSDF &bsdf, CoatedConductorBxDF &coated_conductor_bxdf, SampledWavelengths &lambda,
    const MaterialEvalContext &material_eval_context) const {
    coated_conductor_bxdf = material->get_coated_conductor_bsdf(material_eval_context, lambda);
    bsdf.init_bxdf(&coated_conductor_bxdf);
}

PBRT_GPU
void SurfaceInteraction::init_coated_diffuse_bsdf(
    BSDF &bsdf, CoatedDiffuseBxDF &coated_diffuse_bxdf, SampledWavelengths &lambda,
    const MaterialEvalContext &material_eval_context) const {
    coated_diffuse_bxdf = material->get_coated_diffuse_bsdf(material_eval_context, lambda);
    bsdf.init_bxdf(&coated_diffuse_bxdf);
}

PBRT_GPU
void SurfaceInteraction::init_conductor_bsdf(
    BSDF &bsdf, ConductorBxDF &conductor_bxdf, SampledWavelengths &lambda,
    const MaterialEvalContext &material_eval_context) const {
    conductor_bxdf = material->get_conductor_bsdf(material_eval_context, lambda);
    bsdf.init_bxdf(&conductor_bxdf);
}

PBRT_GPU
void SurfaceInteraction::init_dielectric_bsdf(
    BSDF &bsdf, DielectricBxDF &dielectric_bxdf, SampledWavelengths &lambda,
    const MaterialEvalContext &material_eval_context) const {
    dielectric_bxdf = material->get_dielectric_bsdf(material_eval_context, lambda);
    bsdf.init_bxdf(&dielectric_bxdf);
}

PBRT_GPU
void SurfaceInteraction::init_diffuse_bsdf(BSDF &bsdf, DiffuseBxDF &diffuse_bxdf,
                                           SampledWavelengths &lambda,
                                           const MaterialEvalContext &material_eval_context) const {
    diffuse_bxdf = material->get_diffuse_bsdf(material_eval_context, lambda);
    bsdf.init_bxdf(&diffuse_bxdf);
}

PBRT_GPU
SampledSpectrum SurfaceInteraction::le(const Vector3f w, const SampledWavelengths &lambda) const {
    if (area_light == nullptr) {
        return SampledSpectrum(0.0);
    }

    return area_light->l(p(), n, uv, w, lambda);
}
