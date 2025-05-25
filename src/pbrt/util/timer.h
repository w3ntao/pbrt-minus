#pragma once
#include <chrono>
#include <map>
#include <pbrt/gpu/macro.h>
#include <string>

class Timer {
  public:
    Timer() {
        time_zero = std::chrono::system_clock::now();
        click = time_zero;
    }

    void start() {
        click = std::chrono::system_clock::now();
    }

    void stop(const std::string &tag) {
        const std::chrono::duration<Real> duration = std::chrono::system_clock::now() - click;

        click = std::chrono::system_clock::now();

        if (recorder.find(tag) == recorder.end()) {
            recorder[tag] = duration.count();
            return;
        }

        recorder[tag] += duration.count();
    }

    void print() const {
        const Real all_time = std::chrono::duration<Real>({click - time_zero}).count();

        printf("total time: %.2f\n", all_time);

        Real examine_time = 0;
        for (auto const &[tag, value] : recorder) {
            printf("    %s: %.2f sec (%.1f%)\n", tag.c_str(), value, value * 100.f / all_time);

            examine_time += value;
        }

        constexpr auto tolerance = 0.001;
        if (const auto error_rate = std::abs(examine_time - all_time) / all_time;
            error_rate > tolerance) {
            printf("error rate: %f > %f\n", error_rate, tolerance);

            REPORT_FATAL_ERROR();
        }
    }

  private:
    std::map<std::string, Real> recorder;

    std::chrono::time_point<std::chrono::system_clock> time_zero;
    std::chrono::time_point<std::chrono::system_clock> click;
};
