#pragma once

#include "pbrt/euclidean_space/normal3f.h"
#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/point3.h"
#include "pbrt/util/macro.h"

class Interaction;
class SurfaceInteraction;

class SurfaceInteraction;

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
    explicit TextureEvalContext(const Interaction &intr);

    PBRT_CPU_GPU
    explicit TextureEvalContext(const SurfaceInteraction &si);
};

struct MaterialEvalContext : public TextureEvalContext {
    // MaterialEvalContext Public Methods
    MaterialEvalContext() = default;

    PBRT_CPU_GPU
    MaterialEvalContext(const SurfaceInteraction &si);

    Vector3f wo;
    Normal3f ns;
    Vector3f dpdus;
};
