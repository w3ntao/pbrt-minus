#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

class ThreadPool {
  public:
    ThreadPool(const int num_threads = std::thread::hardware_concurrency()) {
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([this] {
                while (true) {
                    std::function<void()> next_job;
                    {
                        std::unique_lock lock(mtx);
                        cv.wait(lock, [this] { return !job_queue.empty() || quit; });

                        if (job_queue.empty()) {
                            if (quit) {
                                return;
                            }
                            continue;
                        }

                        next_job = std::move(job_queue.front());
                        job_queue.pop();
                    }

                    cv.notify_all();
                    next_job();

                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        num_active_jobs -= 1;
                    }

                    cv.notify_all();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock lock(mtx);
            quit = true;
        }

        cv.notify_all();

        for (auto &t : threads) {
            t.join();
        }
    }

    void parallel_execute(const int start, const int end,
                          const std::function<void(int)> &function_ptr) {
        {
            std::unique_lock lock(mtx);
            for (int i = start; i < end; ++i) {
                job_queue.emplace(std::move([i, &function_ptr] { function_ptr(i); }));

                num_active_jobs += 1;
            }
        }
        cv.notify_all();

        while (true) {
            std::unique_lock lock(mtx);
            cv.wait(lock, [this] { return num_active_jobs == 0; });

            break;
        }
    }

    void submit(const std::function<void()> &function_ptr) {
        std::unique_lock lock(mtx);
        job_queue.emplace(std::move([function_ptr] { function_ptr(); }));
        num_active_jobs += 1;
        cv.notify_all();
    }

    void sync() {
        while (true) {
            std::unique_lock lock(mtx);
            cv.wait(lock, [this] { return num_active_jobs == 0; });
            break;
        }
    }

  private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> job_queue;
    std::mutex mtx;
    std::condition_variable cv;

    int num_active_jobs = 0;
    bool quit = false;
};
