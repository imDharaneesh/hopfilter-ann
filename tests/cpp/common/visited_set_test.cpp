#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/visited_set.hpp>

namespace hopfilter {
namespace {

TEST(VisitedSetTest, MarkReturnsTrueOnlyOnFirstVisit) {
    VisitedSet visited{4};

    EXPECT_TRUE(visited.markVisited(2));
    EXPECT_FALSE(visited.markVisited(2));
    EXPECT_TRUE(visited.isVisited(2));
    EXPECT_FALSE(visited.isVisited(1));
}

TEST(VisitedSetTest, ClearForgetsPreviousVisits) {
    VisitedSet visited{4};
    static_cast<void>(visited.markVisited(0));
    static_cast<void>(visited.markVisited(3));

    visited.clear();

    EXPECT_FALSE(visited.isVisited(0));
    EXPECT_FALSE(visited.isVisited(3));
    EXPECT_TRUE(visited.markVisited(0));
}

TEST(VisitedSetTest, EpochWrapAroundClearsStaleStamps) {
    VisitedSet visited{3, std::numeric_limits<std::uint32_t>::max()};
    static_cast<void>(visited.markVisited(1));

    visited.clear();

    EXPECT_FALSE(visited.isVisited(1));
    static_cast<void>(visited.markVisited(2));
    visited.clear();
    EXPECT_FALSE(visited.isVisited(2));
    EXPECT_FALSE(visited.isVisited(1));
}

TEST(VisitedSetPoolTest, ReusesReleasedSetsAndClearsThem) {
    VisitedSetPool pool{5};
    const VisitedSet* first = nullptr;
    {
        const auto lease = pool.acquire();
        first = &lease.get();
        static_cast<void>(lease.get().markVisited(4));
    }
    EXPECT_EQ(pool.idleCount(), 1U);

    const auto lease = pool.acquire();

    EXPECT_EQ(&lease.get(), first);
    EXPECT_FALSE(lease.get().isVisited(4));
    EXPECT_EQ(pool.idleCount(), 0U);
}

TEST(VisitedSetPoolTest, ConcurrentLeasesGetDistinctSets) {
    VisitedSetPool pool{8};
    const auto first = pool.acquire();
    const auto second = pool.acquire();

    EXPECT_NE(&first.get(), &second.get());
    EXPECT_EQ(first.get().capacity(), 8U);
}

constexpr std::size_t kPoolCapacity = 64;

void leaseRepeatedly(VisitedSetPool& pool) {
    constexpr int kRounds = 200;
    for (int round = 0; round < kRounds; ++round) {
        const auto lease = pool.acquire();
        for (NodeId id = 0; id < kPoolCapacity; ++id) {
            static_cast<void>(lease.get().markVisited(id));
        }
    }
}

TEST(VisitedSetPoolTest, ManyThreadsCanAcquireAndRelease) {
    constexpr std::size_t kThreads = 8;
    VisitedSetPool pool{kPoolCapacity};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (std::size_t worker = 0; worker < kThreads; ++worker) {
        workers.emplace_back(leaseRepeatedly, std::ref(pool));
    }
    for (auto& worker : workers) {
        worker.join();
    }

    EXPECT_LE(pool.idleCount(), kThreads);
    EXPECT_GE(pool.idleCount(), 1U);
}

}  // namespace
}  // namespace hopfilter
