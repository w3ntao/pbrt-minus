#pragma once

#include <pbrt/euclidean_space/normal3f.h>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/euclidean_space/point3.h>

class Interaction;
class SurfaceInteraction;

struct TextureEvalContext {
    Point3f p = Point3f(NAN, NAN, NAN);
    Vector3f dpdx;
    Vector3f dpdy;

    Normal3f n;
    Point2f uv;

    Real dudx = 0;
    Real dudy = 0;
    Real dvdx = 0;
    Real dvdy = 0;

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
