#pragma once

#include "pbrt/util/macro.h"

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/point3.h"
#include "pbrt/euclidean_space/vector3.h"
#include "pbrt/euclidean_space/normal3f.h"

#include "pbrt/base/interaction.h"

struct TextureEvalContext {
    Point3f p;
    Vector3f dpdx;
    Vector3f dpdy;

    Normal3f n;
    Point2f uv;

    FloatType dudx = 0;
    FloatType dudy = 0;
    FloatType dvdx = 0;
    FloatType dvdy = 0;

    int faceIndex = 0;

    TextureEvalContext() = default;

    PBRT_CPU_GPU
    explicit TextureEvalContext(const Interaction &intr) : p(intr.pi.to_point3f()), uv(intr.uv) {}

    PBRT_CPU_GPU
    explicit TextureEvalContext(const SurfaceInteraction &si)
        : p(si.pi.to_point3f()), dpdx(si.dpdx), dpdy(si.dpdy), n(si.n), uv(si.uv), dudx(si.dudx),
          dudy(si.dudy), dvdx(si.dvdx), dvdy(si.dvdy), faceIndex(si.faceIndex) {}
};
