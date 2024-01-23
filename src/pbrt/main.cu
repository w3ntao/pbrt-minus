#include "pbrt/base/renderer.cuh"

using namespace std;

int main() {
    int num_samples = 1;

    render(num_samples, "output.png");

    return 0;
}
