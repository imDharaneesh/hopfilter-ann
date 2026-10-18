#include <algorithm>
#include <cstddef>
#include <memory>
#include <set>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/distance/metric_registry.hpp>
#include <hopfilter/index/hnsw_index.hpp>
#include <hopfilter/index/hnsw_index_builder.hpp>
#include <hopfilter/index/index_config.hpp>

#include "support/brute_force.hpp"
#include "support/graph_walk.hpp"
#include "support/random_vectors.hpp"

namespace hopfilter {
namespace {

constexpr std::size_t kDimension = 16;
constexpr std::size_t kPoints = 4000;
constexpr std::size_t kThreads = 8;
constexpr std::size_t kNeighbors = 10;
constexpr std::size_t kMaxUnreachableNodes = kPoints / 1000;

IndexConfig configFor(std::size_t capacity) {
    IndexConfig config;
    config.dimension = kDimension;
    config.capacity = capacity;
    config.maxNeighbors = 12;
    config.efConstruction = 100;
    return config;
}

std::vector<float> flatten(const std::vector<std::vector<float>>& rows) {
    std::vector<float> flat;
    flat.reserve(rows.size() * kDimension);
    for (const auto& row : rows) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

std::unique_ptr<HnswIndex> buildInParallel(const std::vector<std::vector<float>>& data,
                                           std::size_t threads) {
    auto index = HnswIndexBuilder{}.withConfig(configFor(data.size())).build();
    static_cast<void>(index->insertBatch(flatten(data), threads));
    return index;
}

double meanRecall(const HnswIndex& index,
                  const std::vector<std::vector<float>>& data,
                  const std::vector<std::vector<float>>& queries) {
    const auto metric = MetricRegistry::withBuiltins().create("l2");
    double total = 0.0;
    for (const auto& query : queries) {
        const auto truth = testing::bruteForceTopK(data, query, kNeighbors, *metric);
        total += testing::recall(index.search(query, kNeighbors, SearchRequest{64}), truth);
    }
    return total / static_cast<double>(queries.size());
}

void expectNoSelfLoopsOrDuplicates(const HnswGraph& graph) {
    for (NodeId id = 0; id < graph.size(); ++id) {
        for (LevelIndex level = 0; level <= graph.levelOf(id); ++level) {
            const auto neighbors = graph.neighbors(id, level);
            const std::unordered_set<NodeId> unique(neighbors.begin(), neighbors.end());
            ASSERT_EQ(unique.size(), neighbors.size()) << "duplicate edge at " << id;
            ASSERT_FALSE(unique.contains(id)) << "self loop at " << id;
            ASSERT_LE(neighbors.size(), graph.maxNeighbors(level));
        }
    }
}

TEST(HnswParallelBuildTest, BatchIdsFollowRowOrder) {
    testing::RandomVectors random{1};
    const auto data = random.vectors(50, kDimension);
    auto index = HnswIndexBuilder{}.withConfig(configFor(100)).build();

    const NodeId first = index->insertBatch(flatten(data), kThreads);
    const NodeId second = index->insertBatch(flatten(random.vectors(10, kDimension)), kThreads);

    EXPECT_EQ(first, 0U);
    EXPECT_EQ(second, 50U);
    for (NodeId id = 0; id < data.size(); ++id) {
        EXPECT_TRUE(std::ranges::equal(index->vectors().at(id), data[id]));
    }
}

TEST(HnswParallelBuildTest, OversizedBatchInsertsNothing) {
    testing::RandomVectors random{2};
    auto index = HnswIndexBuilder{}.withConfig(configFor(10)).build();
    static_cast<void>(index->insertBatch(flatten(random.vectors(6, kDimension)), 1));

    EXPECT_THROW(static_cast<void>(index->insertBatch(flatten(random.vectors(5, kDimension)), 2)),
                 IndexStateError);
    EXPECT_EQ(index->size(), 6U);
}

TEST(HnswParallelBuildTest, RejectsPartialRowsAndAcceptsEmptyBatches) {
    auto index = HnswIndexBuilder{}.withConfig(configFor(10)).build();
    const std::vector<float> partial(kDimension + 3, 0.0F);

    EXPECT_THROW(static_cast<void>(index->insertBatch(partial, 2)), InvalidArgumentError);
    EXPECT_EQ(index->insertBatch(std::span<const float>{}, 2), 0U);
    EXPECT_EQ(index->size(), 0U);
}

TEST(HnswParallelBuildTest, SingleThreadBatchBuildsTheSameGraphAsSequentialInserts) {
    testing::RandomVectors random{3};
    const auto data = random.vectors(500, kDimension);
    const auto sequential = testing::buildIndex(data, configFor(data.size()));
    const auto batched = buildInParallel(data, 1);

    for (NodeId id = 0; id < data.size(); ++id) {
        ASSERT_EQ(batched->graph().levelOf(id), sequential->graph().levelOf(id));
        for (LevelIndex level = 0; level <= sequential->graph().levelOf(id); ++level) {
            ASSERT_TRUE(std::ranges::equal(batched->graph().neighbors(id, level),
                                           sequential->graph().neighbors(id, level)));
        }
    }
}

TEST(HnswParallelBuildTest, ParallelBuildKeepsRecallInvariantsAndReachability) {
    testing::RandomVectors random{4};
    const auto data = random.vectors(kPoints, kDimension);
    const auto queries = random.vectors(200, kDimension);

    const auto index = buildInParallel(data, kThreads);

    EXPECT_GE(meanRecall(*index, data, queries), 0.95);
    expectNoSelfLoopsOrDuplicates(index->graph());
    const auto reached = testing::reachableOnLevelZero(index->graph());
    const auto unreachable = kPoints - static_cast<std::size_t>(std::ranges::count(reached, true));
    EXPECT_LE(unreachable, kMaxUnreachableNodes);
}

TEST(HnswParallelBuildTest, ConcurrentSingleInsertsAssignEveryId) {
    testing::RandomVectors random{5};
    const auto data = random.vectors(1000, kDimension);
    auto index = HnswIndexBuilder{}.withConfig(configFor(data.size())).build();
    std::vector<std::vector<NodeId>> idsByThread(kThreads);

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (std::size_t worker = 0; worker < kThreads; ++worker) {
        workers.emplace_back([&, worker] {
            for (std::size_t row = worker; row < data.size(); row += kThreads) {
                idsByThread[worker].push_back(index->insert(data[row]));
            }
        });
    }
    for (auto& thread : workers) {
        thread.join();
    }

    std::set<NodeId> ids;
    for (const auto& assigned : idsByThread) {
        ids.insert(assigned.begin(), assigned.end());
    }
    EXPECT_EQ(ids.size(), data.size());
    expectNoSelfLoopsOrDuplicates(index->graph());
}

}
}
