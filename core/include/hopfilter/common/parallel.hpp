#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace hopfilter {

/// Number of worker threads to use when the caller asks for ``requested`` (0 = all cores).
[[nodiscard]] inline std::size_t resolveThreadCount(std::size_t requested) noexcept {
    if (requested > 0) {
        return requested;
    }
    return std::max<std::size_t>(1, std::thread::hardware_concurrency());
}

/// Keeps the first exception thrown by any worker of a parallel loop.
class FirstError {
public:
    void capture() noexcept {
        const std::scoped_lock lock{mutex_};
        if (!error_) {
            error_ = std::current_exception();
        }
        failed_ = true;
    }

    [[nodiscard]] bool failed() const noexcept { return failed_; }

    void rethrowIfAny() const {
        if (error_) {
            std::rethrow_exception(error_);
        }
    }

private:
    std::mutex mutex_;
    std::exception_ptr error_;
    std::atomic<bool> failed_{false};
};

/// Calls ``body(index)`` for every index in ``[0, count)`` on up to ``threads`` threads
/// (0 = all cores). Indices are handed out dynamically, so uneven work balances itself.
/// The first exception thrown by any call stops new work and is rethrown here once all
/// threads have finished.
template <typename Body>
void parallelFor(std::size_t count, std::size_t threads, const Body& body) {
    const std::size_t workers = std::min(resolveThreadCount(threads), count);
    std::atomic<std::size_t> next{0};
    FirstError errors;
    const auto work = [&] {
        for (std::size_t index = next++; index < count && !errors.failed(); index = next++) {
            try {
                body(index);
            } catch (...) {
                errors.capture();
            }
        }
    };
    {
        std::vector<std::jthread> pool;
        pool.reserve(workers > 0 ? workers - 1 : 0);
        for (std::size_t worker = 1; worker < workers; ++worker) {
            pool.emplace_back(work);
        }
        work();
    }
    errors.rethrowIfAny();
}

}
