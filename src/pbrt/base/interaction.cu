#include <pbrt/base/camera.h>
#include <pbrt/base/interaction.h>
#include <pbrt/base/light.h>
#include <pbrt/base/material.h>
#include <pbrt/textures/texture_eval_context.h>

PBRT_CPU_GPU
void SurfaceInteraction::compute_differentials(const Camera *camera, uint samples_per_pixel) {
    // different with PBRT-v4: ignore the DifferentialRay
    camera->approximate_dp_dxy(p(), n, samples_per_pixel, &dpdx, &dpdy);

    // Estimate screen-space change in $(u,v)$
    // Compute $\transpose{\XFORM{A}} \XFORM{A}$ and its determinant
    Real ata00 = dpdu.dot(dpdu);
    Real ata01 = dpdu.dot(dpdv);
    Real ata11 = dpdv.dot(dpdv);

    Real invDet = 1.0 / difference_of_products(ata00, ata11, ata01, ata01);
    invDet = std::isfinite(invDet) ? invDet : 0.0;

    // Compute $\transpose{\XFORM{A}} \VEC{b}$ for $x$ and $y$
    Real atb0x = dpdu.dot(dpdx);
    Real atb1x = dpdv.dot(dpdx);
    Real atb0y = dpdu.dot(dpdy);
    Real atb1y = dpdv.dot(dpdy);

    // Compute $u$ and $v$ derivatives with respect to $x$ and $y$
    dudx = difference_of_products(ata11, atb0x, ata01, atb1x) * invDet;
    dvdx = difference_of_products(ata00, atb1x, ata01, atb0x) * invDet;
    dudy = difference_of_products(ata11, atb0y, ata01, atb1y) * invDet;
    dvdy = difference_of_products(ata00, atb1y, ata01, atb0y) * invDet;

    // Clamp derivatives of $u$ and $v$ to reasonable values
    dudx = std::isfinite(dudx) ? clamp<Real>(dudx, -1e8f, 1e8f) : 0.0;
    dvdx = std::isfinite(dvdx) ? clamp<Real>(dvdx, -1e8f, 1e8f) : 0.0;
    dudy = std::isfinite(dudy) ? clamp<Real>(dudy, -1e8f, 1e8f) : 0.0;
    dvdy = std::isfinite(dvdy) ? clamp<Real>(dvdy, -1e8f, 1e8f) : 0.0;
}

PBRT_CPU_GPU
void SurfaceInteraction::set_intersection_properties(const Material *_material,
                                                     const Light *_area_light) {
    if (_material->get_material_type() == Material::Type::mix) {
        const auto u = pbrt::hash_float(pi, wo);
        material = _material->get_material_from_mix_material(u);
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

PBRT_CPU_GPU
BSDF SurfaceInteraction::get_bsdf(SampledWavelengths &lambda, const Camera *camera,
                                  uint samples_per_pixel) {
    compute_differentials(camera, samples_per_pixel);

    auto material_eval_context = MaterialEvalContext(*this);

    BSDF bsdf(material_eval_context.ns, material_eval_context.dpdus);

    bsdf.init_bxdf(material, lambda, material_eval_context);

    return bsdf;
}

PBRT_CPU_GPU
SampledSpectrum SurfaceInteraction::le(const Vector3f w, const SampledWavelengths &lambda) const {
    if (area_light == nullptr) {
        return SampledSpectrum(0.0);
    }

    return area_light->l(p(), n, uv, w, lambda);
}
