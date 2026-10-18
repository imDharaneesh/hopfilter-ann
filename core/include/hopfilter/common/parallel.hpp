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

/// Calls ``body(index)`` for every index in ``[0, count)`` on up to ``threads`` threads
/// (0 = all cores). Indices are handed out dynamically, so uneven work balances itself.
/// The first exception thrown by any call stops new work and is rethrown here once all
/// threads have finished.
template <typename Body>
void parallelFor(std::size_t count, std::size_t threads, const Body& body) {
    const std::size_t workers = std::min(resolveThreadCount(threads), count);
    if (workers <= 1) {
        for (std::size_t index = 0; index < count; ++index) {
            body(index);
        }
        return;
    }
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr firstError;
    std::mutex errorMutex;
    const auto work = [&] {
        for (std::size_t index = next++; index < count && !failed; index = next++) {
            try {
                body(index);
            } catch (...) {
                const std::scoped_lock lock{errorMutex};
                firstError = firstError ? firstError : std::current_exception();
                failed = true;
            }
        }
    };
    std::vector<std::jthread> pool;
    pool.reserve(workers - 1);
    for (std::size_t worker = 1; worker < workers; ++worker) {
        pool.emplace_back(work);
    }
    work();
    pool.clear();
    if (firstError) {
        std::rethrow_exception(firstError);
    }
}

}
