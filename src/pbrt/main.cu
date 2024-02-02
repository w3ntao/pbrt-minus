#include "pbrt/scene/builder.h"

using namespace std;

int main(int argc, const char **argv) {
    const auto command_line_option = CommandLineOption(argc, argv);
    SceneBuilder::render_pbrt(command_line_option);

    return 0;
}
