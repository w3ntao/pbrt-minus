#include "pbrt/scene/builder.h"

using namespace std;

int main() {
    std::string input_file = "/home/wentao/Desktop/pbrt-minus-scenes/cornell-box/cornell-box.pbrt";
    SceneBuilder::render_pbrt(input_file);
}
