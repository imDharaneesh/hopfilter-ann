#include <cstddef>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/parallel.hpp>
#include <hopfilter/graph/hnsw_graph.hpp>
#include <hopfilter/graph/node_locks.hpp>

namespace hopfilter {
namespace {

TEST(NodeLocksTest, NodesMapToStripesByModulo) {
    const NodeLocks locks{4};

    EXPECT_EQ(locks.stripes(), 4U);
    EXPECT_EQ(locks.stripeOf(1), locks.stripeOf(5));
    EXPECT_NE(locks.stripeOf(1), locks.stripeOf(2));
}

TEST(NodeLocksTest, DifferentStripesDoNotBlockEachOther) {
    const NodeLocks locks{4};
    const auto first = locks.lock(0);

    bool acquired = false;
    std::thread other([&] { acquired = locks.lock(1).owns_lock(); });
    other.join();

    EXPECT_TRUE(acquired);
}

TEST(NodeLocksTest, RejectsZeroStripes) {
    EXPECT_THROW(NodeLocks{0}, InvalidArgumentError);
}

TEST(HnswGraphConcurrencyTest, GuardedWritesToDifferentNodesDoNotInterfere) {
    constexpr std::size_t kNodes = 2'000;
    HnswGraph graph{kNodes, 8, 4};
    for (std::size_t node = 0; node < kNodes; ++node) {
        static_cast<void>(graph.addNode(0));
    }
    const NodeLocks locks{64};

    parallelFor(kNodes, 8, [&](std::size_t node) {
        const auto id = static_cast<NodeId>(node);
        const std::vector<NodeId> neighbors{static_cast<NodeId>((node + 1) % kNodes)};
        const auto guard = locks.lock(id);
        graph.setNeighbors(id, 0, neighbors);
    });

    for (std::size_t node = 0; node < kNodes; ++node) {
        const auto neighbors = graph.neighbors(static_cast<NodeId>(node), 0);
        ASSERT_EQ(neighbors.size(), 1U);
        EXPECT_EQ(neighbors.front(), (node + 1) % kNodes);
    }
}

TEST(HnswGraphConcurrencyTest, AddingNodesNeverMovesExistingNeighborLists) {
    HnswGraph graph{100, 4, 2};
    static_cast<void>(graph.addNode(2));
    graph.setNeighbors(0, 1, std::vector<NodeId>{0});
    const NodeId* before = graph.neighbors(0, 1).data();

    for (int node = 1; node < 100; ++node) {
        static_cast<void>(graph.addNode(2));
    }

    EXPECT_EQ(graph.neighbors(0, 1).data(), before);
    EXPECT_EQ(graph.size(), 100U);
}

}
}
