#include "pbrt/scene/builder.h"
#include "pbrt/base/renderer.cuh"

using namespace std;

int main() {
    std::string filename =
        "/home/wentao/Desktop/pbrt-minus-scenes/killeroos/killeroo-wall-nolight.pbrt";
    auto scene_builder = SceneBuilder::read_pbrt(filename);

    return 0;

    int num_samples = 1;

    render(num_samples, "output.png");

    return 0;
}
