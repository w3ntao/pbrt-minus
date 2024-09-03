#pragma once

#include "ext/rply/rply.h"
#include "pbrt/euclidean_space/normal3f.h"
#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/point3.h"
#include "pbrt/util/macro.h"
#include <vector>

struct TriQuadMesh {
    std::vector<Point3f> p;
    std::vector<Normal3f> n;
    std::vector<Point2f> uv;
    std::vector<int> faceIndices;
    std::vector<int> triIndices;
    std::vector<int> quadIndices;

    static TriQuadMesh read_ply(const std::string &filename);
};
