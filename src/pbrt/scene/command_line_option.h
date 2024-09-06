#pragma once

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

struct CommandLineOption {
    std::string input_file;
    std::optional<std::string> integrator_name;
    std::string output_file;
    std::optional<int> samples_per_pixel;
    bool megakernel = false;

    CommandLineOption(int argc, const char **argv) {
        int idx = 1;
        while (idx < argc) {
            std::string argument = argv[idx];
            if (argument.size() > 2 && argument.substr(0, 2) == "--") {
                if (argument == "--spp") {
                    samples_per_pixel = stoi(std::string(argv[idx + 1]));
                    idx += 2;
                    continue;
                }

                if (argument == "--integrator") {
                    integrator_name = argv[idx + 1];
                    idx += 2;
                    continue;
                }

                if (argument == "--megakernel") {
                    megakernel = true;
                    idx += 1;
                    continue;
                }

                if (argument == "--output") {
                    output_file = argv[idx + 1];
                    idx += 2;
                    continue;
                }

                const std::string error = "CommandLineOption(): unknown arg: `" + argument + "`";
                throw std::runtime_error(error.c_str());
            }

            input_file = argument;
            idx += 1;
        }

        if (input_file.empty()) {
            std::cout << "please provide a PBRT input file from command line:\n"
                      << "$ pbrt-minus example.pbrt\n";
            exit(1);
        }
    }
};
