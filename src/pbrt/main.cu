#include "pbrt/scene/parser.h"
#include "pbrt/base/renderer.cuh"

using namespace std;

int main() {
    std::string filename =
        "/home/wentao/Desktop/pbrt-minus-scenes/killeroos/killeroo-wall-simple.pbrt";

    auto tokens = parse_pbrt_into_token(filename);

    for (const auto &t : tokens) {
        std::cout << parsing_type_to_string(t.type);
        if (t.value.size() > 0) {
            std::cout << ": { ";
            for (const auto &x : t.value) {
                std::cout << x << " ";
            }
            std::cout << "}";
        }
        std::cout << "\n";
    }

    return 0;

    int num_samples = 1;

    render(num_samples, "output.png");

    return 0;
}
