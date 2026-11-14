#pragma once

#include <array>
#include <cstdint>

namespace heatmap {

// sRGB palette: https://waldyrious.net/viridis-palette-generator/
inline constexpr std::array<std::array<uint8_t, 3>, 7> viridis_srgb = {{
    {68, 1, 84}, {68, 57, 131}, {49, 104, 142}, {33, 145, 140},
    {53, 183, 121}, {144, 215, 67}, {253, 231, 37},
}};

} // namespace heatmap
