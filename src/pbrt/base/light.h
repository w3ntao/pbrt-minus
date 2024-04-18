#pragma once

#include "pbrt/euclidean_space/transform.h"

enum class LightType {
    delta_position,
    delta_direction,
    area,
    infinite,
};

struct LightBase {
    LightType light_type;
    Transform render_from_light;
};
