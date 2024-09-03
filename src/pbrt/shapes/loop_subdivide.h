#pragma once

#include "pbrt/euclidean_space/point3.h"
#include <vector>

struct LoopSubdivide {
    std::vector<int> vertex_indices;
    std::vector<Point3f> p_limit;

    LoopSubdivide(int nLevels, const std::vector<int> &vertexIndices,
                  const std::vector<Point3f> &p);
};
