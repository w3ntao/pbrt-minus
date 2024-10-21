#include "pbrt/euclidean_space/frame.h"
#include "pbrt/euclidean_space/transform.h"

PBRT_CPU_GPU
Transform Frame::to_transform() const {
    FloatType array[4][4] = {
        {x.x, x.y, x.z, 0},
        {y.x, y.y, y.z, 0},
        {z.x, z.y, z.z, 0},
        {0, 0, 0, 1},
    };

    return Transform(array);
}
