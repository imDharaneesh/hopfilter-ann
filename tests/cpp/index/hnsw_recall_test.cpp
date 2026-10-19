#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/distance/metric_registry.hpp>
#include <hopfilter/index/hnsw_index.hpp>
#include <hopfilter/index/index_config.hpp>

#include "support/brute_force.hpp"
#include "support/graph_walk.hpp"
#include "support/random_vectors.hpp"

namespace hopfilter {
namespace {

constexpr std::size_t kPoints = 3000;
constexpr std::size_t kDimension = 16;
constexpr std::size_t kQueries = 200;
constexpr std::size_t kNeighbors = 10;
constexpr std::size_t kEfSearch = 64;

IndexConfig configFor(const std::string& metricName) {
    IndexConfig config;
    config.maxNeighbors = 12;
    config.efConstruction = 100;
    config.metricName = metricName;
    return config;
}

double meanRecall(const HnswIndex& index,
                  const std::vector<std::vector<float>>& data,
                  const std::vector<std::vector<float>>& queries) {
    const auto metric = MetricRegistry::withBuiltins().create(index.metric().name());
    double total = 0.0;
    for (const auto& query : queries) {
        const auto truth = testing::bruteForceTopK(data, query, kNeighbors, *metric);
        total += testing::recall(index.search(query, kNeighbors, SearchRequest{kEfSearch}), truth);
    }
    return total / static_cast<double>(queries.size());
}

class RecallTest : public ::testing::TestWithParam<std::string> {};

TEST_P(RecallTest, MatchesBruteForceOnRandomData) {
    testing::RandomVectors random{17};
    const auto data = random.vectors(kPoints, kDimension);
    const auto queries = random.vectors(kQueries, kDimension);
    const auto index = testing::buildIndex(data, configFor(GetParam()));

    EXPECT_GE(meanRecall(*index, data, queries), 0.95) << "metric " << GetParam();
}

INSTANTIATE_TEST_SUITE_P(Metrics, RecallTest, ::testing::Values("l2", "ip", "cosine"));

class GraphInvariantsTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        testing::RandomVectors random{23};
        sharedData = random.vectors(kPoints, kDimension);
        sharedIndex = testing::buildIndex(sharedData, configFor("l2"));
    }

    static void TearDownTestSuite() { sharedIndex.reset(); }

    static const HnswGraph& graph() { return sharedIndex->graph(); }

    static SearchResults searchFor(std::size_t query) {
        return sharedIndex->search(sharedData[query], kNeighbors, SearchRequest{});
    }

    static inline std::vector<std::vector<float>> sharedData;
    static inline std::unique_ptr<HnswIndex> sharedIndex;
};

void expectValidNeighborList(const HnswGraph& graph, NodeId id, LevelIndex level) {
    const auto neighbors = graph.neighbors(id, level);
    const std::unordered_set<NodeId> unique(neighbors.begin(), neighbors.end());
    EXPECT_LE(neighbors.size(), graph.maxNeighbors(level));
    EXPECT_EQ(unique.size(), neighbors.size()) << "duplicate neighbor of " << id;
    EXPECT_FALSE(unique.contains(id)) << "self loop at " << id;
    for (const NodeId neighbor : neighbors) {
        EXPECT_GE(graph.levelOf(neighbor), level);
    }
}

TEST_F(GraphInvariantsTest, NeighborListsRespectDegreeAndLevelBounds) {
    for (NodeId id = 0; id < graph().size(); ++id) {
        for (LevelIndex level = 0; level <= graph().levelOf(id); ++level) {
            expectValidNeighborList(graph(), id, level);
        }
    }
}

TEST_F(GraphInvariantsTest, EntryPointIsOnTheTopLevel) {
    const auto entryPoint = graph().entryPoint();
    if (!entryPoint) {
        FAIL() << "the graph has no entry point";
    }
    EXPECT_EQ(graph().levelOf(*entryPoint), graph().maxLevel());
    for (NodeId id = 0; id < graph().size(); ++id) {
        EXPECT_LE(graph().levelOf(id), graph().maxLevel());
    }
}

TEST_F(GraphInvariantsTest, EveryNodeIsReachableOnLevelZero) {
    const auto reached = testing::reachableOnLevelZero(graph());

    EXPECT_EQ(std::ranges::count(reached, true), static_cast<std::ptrdiff_t>(graph().size()));
}

std::vector<SearchResults> searchAll() {
    std::vector<SearchResults> results;
    results.reserve(kQueries);
    for (std::size_t query = 0; query < kQueries; ++query) {
        results.push_back(GraphInvariantsTest::searchFor(query));
    }
    return results;
}

TEST_F(GraphInvariantsTest, ConcurrentSearchesMatchSequentialResults) {
    constexpr std::size_t kThreads = 4;
    const auto expected = searchAll();
    std::vector<std::vector<SearchResults>> actual(kThreads);
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (std::size_t worker = 0; worker < kThreads; ++worker) {
        workers.emplace_back([&actual, worker] { actual[worker] = searchAll(); });
    }
    for (auto& thread : workers) {
        thread.join();
    }

    for (const auto& results : actual) {
        EXPECT_EQ(results, expected);
    }
}

}
}
