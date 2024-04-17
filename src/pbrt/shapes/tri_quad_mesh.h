#pragma once

#include <vector>

#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/point3.h"
#include "pbrt/euclidean_space/normal3f.h"

#include "ext/rply/rply.h"

struct TriQuadMesh {
    std::vector<Point3f> p;
    std::vector<Normal3f> n;
    std::vector<Point2f> uv;
    std::vector<int> faceIndices;
    std::vector<int> triIndices;
    std::vector<int> quadIndices;

    static TriQuadMesh read_ply(const std::string &filename);
};
