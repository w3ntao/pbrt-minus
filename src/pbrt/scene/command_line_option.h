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
    bool preview = false;

    CommandLineOption(int argc, const char **argv) {

        int idx = 1;
        while (idx < argc) {
            std::string argument = argv[idx];

            auto read_next_argument = [&]() -> std::string {
                if (idx + 1 >= argc) {
                    const std::string error =
                        "CommandLineOption(): expect valid argument after `" + argument + "`";
                    throw std::runtime_error(error.c_str());
                }

                return argv[idx + 1];
            };

            if (argument.size() > 2 && argument.substr(0, 2) == "--") {
                if (argument == "--spp") {
                    samples_per_pixel = stoi(read_next_argument());
                    idx += 2;
                    continue;
                }

                if (argument == "--integrator") {
                    integrator_name = read_next_argument();
                    idx += 2;
                    continue;
                }

                if (argument == "--outfile") {
                    output_file = read_next_argument();
                    idx += 2;
                    continue;
                }

                if (argument == "--preview") {
                    preview = true;
                    idx += 1;
                    continue;
                }

                const std::string error = "CommandLineOption(): unknown arg: `" + argument + "`";
                throw std::runtime_error(error.c_str());
            }

            if (std::filesystem::path p(argument); p.extension() == ".pbrt") {
                input_file = argument;
            }

            idx += 1;
        }

        if (input_file.empty()) {
            std::cout << "please provide a PBRT input file from command line:\n"
                      << "$ pbrt-minus example.pbrt\n";
            exit(1);
        }
    }
};
