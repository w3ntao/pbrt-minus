#include "pbrt/base/interaction.h"
#include "pbrt/euclidean_space/transform.h"

PBRT_CPU_GPU
SurfaceInteraction Transform::operator()(const SurfaceInteraction &si) const {
    SurfaceInteraction ret;
    const Transform &t = *this;

    ret.pi = t(si.pi);
    // Transform remaining members of _SurfaceInteraction_

    ret.n = t(si.n).normalize();
    ret.wo = t(si.wo).normalize();

    ret.uv = si.uv;
    ret.dpdu = t(si.dpdu);
    ret.dpdv = t(si.dpdv);

    ret.dndu = t(si.dndu);
    ret.dndv = t(si.dndv);

    ret.shading.n = t(si.shading.n).normalize();

    ret.shading.dpdu = t(si.shading.dpdu);
    ret.shading.dpdv = t(si.shading.dpdv);
    ret.shading.dndu = t(si.shading.dndu);
    ret.shading.dndv = t(si.shading.dndv);

    ret.dudx = si.dudx;
    ret.dvdx = si.dvdx;
    ret.dudy = si.dudy;
    ret.dvdy = si.dvdy;
    ret.dpdx = t(si.dpdx);
    ret.dpdy = t(si.dpdy);

    ret.material = si.material;
    ret.area_light = si.area_light;

    ret.n = ret.n.face_forward(ret.shading.n);
    ret.shading.n = ret.shading.n.face_forward(ret.n);

    ret.faceIndex = si.faceIndex;

    return ret;
}
