#pragma once

#include "euclidean_space/point2.h"
#include "euclidean_space/normal3f.h"
#include "euclidean_space/point3fi.h"

class Interaction {
  public:
    Point3fi pi;
    Vector3f wo;
    Normal3f n;
    Point2f uv;

    PBRT_CPU_GPU
    Interaction(Point3fi pi, Normal3f n, Point2f uv, Vector3f wo)
        : pi(pi), n(n), uv(uv), wo(wo.normalize()) {}
};

class SurfaceInteraction : public Interaction {
  public:
    Vector3f dpdu, dpdv;
    Normal3f dndu, dndv;
    struct {
        Normal3f n;
        Vector3f dpdu, dpdv;
        Normal3f dndu, dndv;
    } shading;
    int faceIndex = 0;

    Vector3f dpdx, dpdy;
    double dudx = NAN;
    double dvdx = NAN;
    double dudy = NAN;
    double dvdy = NAN;

    PBRT_CPU_GPU
    SurfaceInteraction(Point3fi pi, Point2f uv, Vector3f wo, Vector3f dpdu, Vector3f dpdv,
                       Normal3f dndu, Normal3f dndv, bool flipNormal)
        : Interaction(pi, Normal3f(dpdu.cross(dpdv).normalize()), uv, wo), dpdu(dpdu), dpdv(dpdv),
          dndu(dndu), dndv(dndv) {
        // Initialize shading geometry from true geometry
        shading.n = n;
        shading.dpdu = dpdu;
        shading.dpdv = dpdv;
        shading.dndu = dndu;
        shading.dndv = dndv;

        // Adjust normal based on orientation and handedness
        if (flipNormal) {
            n *= -1;
            shading.n *= -1;
        }
    }
};
