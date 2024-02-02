#pragma once

#include <optional>
#include <string>
#include <filesystem>

struct CommandLineOption {
    std::string input_file;
    std::optional<int> samples_per_pixel;
    std::optional<std::string> integrator;

    CommandLineOption(int argc, const char **argv) {
        int idx = 1;
        while (idx < argc) {
            std::string argument = argv[idx];
            if (argument.size() > 2 && argument.substr(0, 2) == "--") {
                /*
                // TODO: add help information
                if (argument == "--help") {}
                */

                if (argument == "--spp") {
                    samples_per_pixel = stoi(std::string(argv[idx + 1]));
                    idx += 2;
                    continue;
                }
                if (argument == "--integrator") {
                    integrator = std::string(argv[idx + 1]);
                    idx += 2;
                    continue;
                }

                const std::string error = "unkown arg: `" + argument + "`";
                throw std::runtime_error(error.c_str());
            }

            input_file = argument;
            idx += 1;
        }
    }
};
