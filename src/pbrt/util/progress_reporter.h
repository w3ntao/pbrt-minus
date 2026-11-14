#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

class ProgressReporter {
  public:
    explicit ProgressReporter(const long long total) : total(total) {}

    ~ProgressReporter() {
        join();
    }

    void increase(const long long n) {
        counter.fetch_add(n, std::memory_order_relaxed);
    }

    void run() {
        if (progress_thread.joinable()) {
            return;
        }

        const auto start = std::chrono::steady_clock::now();
        progress_thread = std::thread([this, start] {
            while (true) {
                const auto completed = counter.load(std::memory_order_relaxed);
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                const double remaining = completed > 0 ? elapsed * (total - completed) / completed : 0;

                const double percentage = (completed >= total ? 1.0 : static_cast<double>(completed) / total) * 100;
                print_progress(percentage, remaining, elapsed);

                if (completed >= total) {
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }

    void join() {
        if (progress_thread.joinable()) {
            progress_thread.join();
        }
    }

    std::atomic<long long> counter{0};
    long long total = 0;

  private:
    std::thread progress_thread;

    static std::string text_remaining_time(const double remaining) {
        const int total_seconds = static_cast<int>(remaining);
        const int minutes = total_seconds / 60;
        const int hours = minutes / 60;

        const int max_hours = 10;

        if (hours >= max_hours) {
            return " > " + std::to_string(max_hours) + "h";
        }

        if (hours > 0) {
            return std::to_string(hours) + "h " + std::to_string(minutes % 60) + "m";
        }

        if (minutes >= 5) {
            return std::to_string(minutes) + "m";
        }

        const int seconds = total_seconds % 60;

        const std::string text = (seconds >= 10 ? "" : " ") + std::to_string(seconds) + "s";

        if (minutes > 0) {
            return std::to_string(minutes) + "m " + text;
        }

        return text;
    }

    static std::string text_elapsed_time(const double elapsed) {
        const int total_seconds = static_cast<int>(elapsed);
        const int minutes = total_seconds / 60;
        const int hours = minutes / 60;
        const int seconds = total_seconds % 60;

        std::string text = (seconds >= 10 ? "" : " ") + std::to_string(seconds) + "s";

        if (minutes > 0) {
            text = std::to_string(minutes % 60) + "m " + text;
        }

        if (hours > 0) {
            text = std::to_string(hours) + "h " + text;
        }

        return text;
    }

    static void print_progress(const double percentage, const double remaining, const double elapsed) {
        const auto space = std::string(10, ' ');

        std::printf("\rrendering: %5.1f%%  #  %s | %s %s", percentage, text_elapsed_time(elapsed).c_str(),
                    text_remaining_time(remaining).c_str(), space.c_str());

        if (percentage >= 100) {
            std::printf("\n");
        }
        std::fflush(stdout);
    }
};
