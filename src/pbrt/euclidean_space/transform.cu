#include <pbrt/base/interaction.h>
#include <pbrt/base/ray.h>
#include <pbrt/euclidean_space/frame.h>
#include <pbrt/euclidean_space/transform.h>

PBRT_CPU_GPU
Transform::Transform(const Frame &frame) {
    FloatType array[4][4] = {
        {frame.x.x, frame.x.y, frame.x.z, 0},
        {frame.y.x, frame.y.y, frame.y.z, 0},
        {frame.z.x, frame.z.y, frame.z.z, 0},
        {0, 0, 0, 1},
    };

    *this = Transform(array);
}

PBRT_CPU_GPU
Ray Transform::operator()(const Ray &r, FloatType *tMax) const {
    Point3fi o = (*this)(Point3fi(r.o));
    Vector3f d = (*this)(r.d);

    // Offset ray origin to edge of error bounds and compute _tMax_
    if (FloatType lengthSquared = d.squared_length(); lengthSquared > 0) {
        FloatType dt = d.abs_dot(o.error()) / lengthSquared;
        o += d * dt;
        if (tMax) {
            *tMax -= dt;
        }
    }

    return Ray(o.to_point3f(), d);
}

PBRT_CPU_GPU
Ray Transform::apply_inverse(const Ray &r, FloatType *tMax) const {
    Point3fi o = apply_inverse(Point3fi(r.o));
    Vector3f d = apply_inverse(r.d);

    // Offset ray origin to edge of error bounds and compute _tMax_
    auto lengthSquared = d.squared_length();

    if (lengthSquared > 0) {
        Vector3f oError(o.x.width() / 2, o.y.width() / 2, o.z.width() / 2);
        auto dt = d.abs_dot(oError) / lengthSquared;

        o += d * dt;
        if (tMax) {
            *tMax -= dt;
        }
    }

    return Ray(o.to_point3f(), d);
}

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
