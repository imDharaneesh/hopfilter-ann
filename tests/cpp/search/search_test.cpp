#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/types.hpp>
#include <hopfilter/common/visited_set.hpp>
#include <hopfilter/distance/metrics.hpp>
#include <hopfilter/graph/hnsw_graph.hpp>
#include <hopfilter/graph/node_distances.hpp>
#include <hopfilter/graph/node_locks.hpp>
#include <hopfilter/search/search_context.hpp>
#include <hopfilter/search/search_layer.hpp>
#include <hopfilter/search/search_strategy.hpp>
#include <hopfilter/storage/vector_store.hpp>

namespace hopfilter {
namespace {

constexpr std::size_t kPoints = 10;
constexpr std::size_t kMaxDegree = 4;
constexpr std::array<NodeId, 3> kExpressNodes{0, 5, 9};

bool isExpress(NodeId id) {
    return std::ranges::find(kExpressNodes, id) != kExpressNodes.end();
}

class LineGraphTest : public ::testing::Test {
protected:
    LineGraphTest()
        : vectors_(1, kPoints),
          graph_(kPoints, kMaxDegree, kMaxDegree),
          distances_(metric_, vectors_),
          visited_(kPoints),
          context_(graph_, distances_, visited_) {
        for (NodeId id = 0; id < kPoints; ++id) {
            static_cast<void>(vectors_.append(std::vector<float>{static_cast<float>(id)}));
            static_cast<void>(graph_.addNode(isExpress(id) ? 1 : 0));
        }
        linkLevelZeroChain();
        graph_.setNeighbors(0, 1, std::vector<NodeId>{5});
        graph_.setNeighbors(5, 1, std::vector<NodeId>{0, 9});
        graph_.setNeighbors(9, 1, std::vector<NodeId>{5});
        graph_.setEntryPoint(0);
    }

    [[nodiscard]] Neighbor entry(NodeId id, float query) const {
        return {id, distances_.toQuery(std::vector<float>{query}, id)};
    }

    [[nodiscard]] const SearchContext& context() const { return context_; }

private:
    void linkLevelZeroChain() {
        for (NodeId id = 0; id < kPoints; ++id) {
            std::vector<NodeId> neighbors;
            if (id > 0) {
                neighbors.push_back(id - 1);
            }
            if (id + 1 < kPoints) {
                neighbors.push_back(id + 1);
            }
            graph_.setNeighbors(id, 0, neighbors);
        }
    }

    L2Metric metric_;
    VectorStore vectors_;
    HnswGraph graph_;
    NodeDistances distances_;
    VisitedSet visited_;
    SearchContext context_;
};

std::vector<NodeId> idsOf(const SearchResults& results) {
    std::vector<NodeId> ids;
    for (const auto& neighbor : results) {
        ids.push_back(neighbor.id);
    }
    return ids;
}

TEST_F(LineGraphTest, SearchLayerWalksFromFarEntryToNearestNodes) {
    const std::vector<float> query{7.2F};
    const Neighbor start = entry(0, 7.2F);

    const auto results = searchLayer(context(), query, {&start, 1}, 3, 0);

    EXPECT_EQ(idsOf(results), (std::vector<NodeId>{7, 8, 6}));
    EXPECT_NEAR(results.front().distance, 0.04F, 1e-5F);
}

TEST_F(LineGraphTest, SearchLayerReturnsAtMostEfSortedWithoutDuplicates) {
    const std::vector<float> query{4.4F};
    const std::vector<Neighbor> starts{entry(0, 4.4F), entry(9, 4.4F), entry(0, 4.4F)};

    const auto results = searchLayer(context(), query, starts, 5, 0);

    EXPECT_EQ(idsOf(results), (std::vector<NodeId>{4, 5, 3, 6, 2}));
    for (std::size_t index = 1; index < results.size(); ++index) {
        EXPECT_LE(results[index - 1].distance, results[index].distance);
    }
}

TEST_F(LineGraphTest, UpperLevelOnlyFollowsExpressLinks) {
    const std::vector<float> query{8.6F};
    const Neighbor start = entry(0, 8.6F);

    const auto results = searchLayer(context(), query, {&start, 1}, 3, 1);

    EXPECT_EQ(idsOf(results), (std::vector<NodeId>{9, 5, 0}));
}

TEST_F(LineGraphTest, GreedyDescendStopsAboveTheRequestedLevel) {
    const std::vector<float> query{8.6F};

    const Neighbor viaExpress = greedyDescend(context(), query, entry(0, 8.6F), 1, 0);
    const Neighbor noDescent = greedyDescend(context(), query, entry(0, 8.6F), 1, 1);

    EXPECT_EQ(viaExpress.id, 9U);
    EXPECT_EQ(noDescent.id, 0U);
}

TEST_F(LineGraphTest, UnfilteredStrategyReturnsKClosest) {
    const std::vector<float> query{2.9F};

    const auto results = UnfilteredStrategy{}.search(context(), query, 3, SearchRequest{8});

    EXPECT_EQ(idsOf(results), (std::vector<NodeId>{3, 2, 4}));
}

TEST_F(LineGraphTest, UnfilteredStrategyWidensEfToAtLeastK) {
    const std::vector<float> query{0.0F};

    const auto results = UnfilteredStrategy{}.search(context(), query, kPoints, SearchRequest{1});

    EXPECT_EQ(results.size(), kPoints);
}

TEST_F(LineGraphTest, LockedTraversalReturnsTheSameResults) {
    const NodeLocks locks{8};
    VisitedSet visited{kPoints};
    const SearchContext locked{context().graph(), context().distances(), visited, &locks};
    const std::vector<float> query{6.6F};

    const auto plain = UnfilteredStrategy{}.search(context(), query, 4, SearchRequest{8});
    const auto guarded = UnfilteredStrategy{}.search(locked, query, 4, SearchRequest{8});

    EXPECT_EQ(guarded, plain);
}

TEST_F(LineGraphTest, ZeroKReturnsNothing) {
    const std::vector<float> query{1.0F};

    EXPECT_TRUE(UnfilteredStrategy{}.search(context(), query, 0, SearchRequest{}).empty());
}

TEST(UnfilteredStrategyTest, EmptyGraphReturnsNothing) {
    const L2Metric metric;
    const VectorStore vectors{1, 1};
    const HnswGraph graph{1, 2, 2};
    const NodeDistances distances{metric, vectors};
    VisitedSet visited{1};
    const SearchContext context{graph, distances, visited};
    const std::vector<float> query{1.0F};

    EXPECT_TRUE(UnfilteredStrategy{}.search(context, query, 5, SearchRequest{}).empty());
}

}
}
