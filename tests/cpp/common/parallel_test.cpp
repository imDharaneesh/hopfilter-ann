#include <atomic>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/parallel.hpp>

namespace hopfilter {
namespace {

TEST(ParallelForTest, VisitsEveryIndexExactlyOnce) {
    constexpr std::size_t kCount = 10'000;
    std::vector<std::atomic<int>> visits(kCount);

    parallelFor(kCount, 8, [&](std::size_t index) { ++visits[index]; });

    for (const auto& count : visits) {
        EXPECT_EQ(count.load(), 1);
    }
}

TEST(ParallelForTest, SingleThreadRunsInOrderOnTheCallingThread) {
    std::vector<std::size_t> order;
    const auto caller = std::this_thread::get_id();
    bool sameThread = true;

    parallelFor(5, 1, [&](std::size_t index) {
        order.push_back(index);
        sameThread = sameThread && std::this_thread::get_id() == caller;
    });

    EXPECT_EQ(order, (std::vector<std::size_t>{0, 1, 2, 3, 4}));
    EXPECT_TRUE(sameThread);
}

TEST(ParallelForTest, EmptyRangeDoesNothing) {
    std::atomic<int> calls{0};

    parallelFor(0, 4, [&](std::size_t) { ++calls; });

    EXPECT_EQ(calls.load(), 0);
}

TEST(ParallelForTest, RethrowsTheFirstExceptionAfterAllThreadsStop) {
    std::atomic<int> calls{0};
    const auto failAtSeven = [&](std::size_t index) {
        ++calls;
        if (index == 7) {
            throw std::runtime_error("index 7 failed");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    };

    try {
        parallelFor(1'000, 4, failAtSeven);
        FAIL() << "expected the worker exception to propagate";
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string{error.what()}, "index 7 failed");
    }
    EXPECT_LT(calls.load(), 1'000);
}

TEST(ParallelForTest, ZeroThreadsMeansAllCores) {
    EXPECT_GE(resolveThreadCount(0), 1U);
    EXPECT_EQ(resolveThreadCount(3), 3U);
}

}
}
