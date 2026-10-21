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

    static void print_progress(const double percentage, const double remaining, const double elapsed) {
        if (percentage <= 0) {
            std::printf("\r");
        } else {
            const auto space = std::string(30, ' ');
            std::printf("\rrendering: %5.1f%%  #  %4.1f s | %4.1f s%s", percentage, elapsed, remaining, space.c_str());
        }

        if (percentage >= 100) {
            std::printf("\n");
        }
        std::fflush(stdout);
    }
};
