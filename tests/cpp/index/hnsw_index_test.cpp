#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/graph/layer_assigner.hpp>
#include <hopfilter/index/hnsw_index_builder.hpp>
#include <hopfilter/index/index_config.hpp>

#include "support/brute_force.hpp"
#include "support/graph_walk.hpp"
#include "support/random_vectors.hpp"

namespace hopfilter {
namespace {

constexpr std::size_t kDimension = 4;

IndexConfig smallConfig(std::size_t capacity) {
    IndexConfig config;
    config.dimension = kDimension;
    config.capacity = capacity;
    config.maxNeighbors = 4;
    config.efConstruction = 16;
    return config;
}

class ScriptedLayerAssigner final : public LayerAssigner {
public:
    explicit ScriptedLayerAssigner(std::vector<LevelIndex> levels) : levels_(std::move(levels)) {}

    [[nodiscard]] LevelIndex nextLevel() override { return levels_.at(next_++); }

private:
    std::vector<LevelIndex> levels_;
    std::size_t next_ = 0;
};

TEST(HnswIndexTest, EmptyIndexReturnsNoResults) {
    const auto index = HnswIndexBuilder{}.withConfig(smallConfig(4)).build();
    const std::vector<float> query(kDimension, 0.0F);

    EXPECT_TRUE(index->search(query, 3, SearchRequest{}).empty());
}

TEST(HnswIndexTest, InsertReturnsSequentialIds) {
    const auto index = HnswIndexBuilder{}.withConfig(smallConfig(3)).build();
    testing::RandomVectors random{1};

    for (NodeId expected = 0; expected < 3; ++expected) {
        EXPECT_EQ(index->insert(random.vector(kDimension)), expected);
    }
    EXPECT_EQ(index->size(), 3U);
}

TEST(HnswIndexTest, KLargerThanSizeReturnsEverythingSorted) {
    testing::RandomVectors random{2};
    const auto data = random.vectors(5, kDimension);
    const auto index = testing::buildIndex(data, smallConfig(5));

    const auto results = index->search(data[2], 50, SearchRequest{});

    ASSERT_EQ(results.size(), 5U);
    EXPECT_EQ(results.front().id, 2U);
    EXPECT_FLOAT_EQ(results.front().distance, 0.0F);
    for (std::size_t position = 1; position < results.size(); ++position) {
        EXPECT_LE(results[position - 1].distance, results[position].distance);
    }
}

TEST(HnswIndexTest, RejectsWrongDimensionZeroKAndFullIndex) {
    const auto index = HnswIndexBuilder{}.withConfig(smallConfig(1)).build();
    const std::vector<float> wrong(kDimension + 1, 0.0F);
    const std::vector<float> right(kDimension, 0.0F);

    EXPECT_THROW(static_cast<void>(index->insert(wrong)), InvalidArgumentError);
    EXPECT_THROW(static_cast<void>(index->search(wrong, 1, SearchRequest{})), InvalidArgumentError);
    static_cast<void>(index->insert(right));
    EXPECT_THROW(static_cast<void>(index->search(right, 0, SearchRequest{})), InvalidArgumentError);
    EXPECT_THROW(static_cast<void>(index->insert(right)), IndexStateError);
}

std::vector<std::vector<float>> withCopies(std::size_t distinct,
                                           std::size_t copies,
                                           std::uint64_t seed) {
    testing::RandomVectors random{seed};
    auto data = random.vectors(distinct, kDimension);
    data.insert(data.end(), copies, random.vector(kDimension));
    return data;
}

TEST(HnswIndexTest, DuplicateClusterNeitherStrandsOtherNodesNorTrapsSearch) {
    constexpr std::size_t kDistinct = 50;
    constexpr std::size_t kCopies = 30;
    const auto data = withCopies(kDistinct, kCopies, 3);
    const auto index = testing::buildIndex(data, smallConfig(data.size()));

    const auto reached = testing::reachableOnLevelZero(index->graph());
    const auto results = index->search(data.back(), kCopies, SearchRequest{128});

    for (NodeId id = 0; id < kDistinct; ++id) {
        EXPECT_TRUE(reached[id]) << "distinct node " << id << " is unreachable";
    }
    ASSERT_EQ(results.size(), kCopies);
    EXPECT_FLOAT_EQ(results.front().distance, 0.0F);
}

TEST(HnswIndexTest, FewerCopiesThanEfConstructionAreAllRetrievable) {
    constexpr std::size_t kDistinct = 60;
    constexpr std::size_t kCopies = 8;
    const auto data = withCopies(kDistinct, kCopies, 6);
    const auto index = testing::buildIndex(data, smallConfig(data.size()));

    const auto results = index->search(data.back(), kCopies, SearchRequest{64});

    ASSERT_EQ(results.size(), kCopies);
    for (const auto& neighbor : results) {
        EXPECT_GE(neighbor.id, kDistinct);
        EXPECT_FLOAT_EQ(neighbor.distance, 0.0F);
    }
}

TEST(HnswIndexTest, EntryPointMovesToFirstNodeOnHighestLevel) {
    HnswIndexBuilder builder;
    builder.withConfig(smallConfig(5))
        .withLayerAssigner(
            std::make_unique<ScriptedLayerAssigner>(std::vector<LevelIndex>{0, 2, 1, 2, 0}));
    const auto index = builder.build();
    testing::RandomVectors random{4};

    for (int insert = 0; insert < 5; ++insert) {
        static_cast<void>(index->insert(random.vector(kDimension)));
    }

    EXPECT_EQ(index->graph().entryPoint(), NodeId{1});
    EXPECT_EQ(index->graph().maxLevel(), 2);
}

TEST(HnswIndexTest, SameSeedAndOrderGiveIdenticalResults) {
    testing::RandomVectors random{5};
    const auto data = random.vectors(300, kDimension);
    const auto first = testing::buildIndex(data, smallConfig(data.size()));
    const auto second = testing::buildIndex(data, smallConfig(data.size()));

    for (std::size_t query = 0; query < 20; ++query) {
        EXPECT_EQ(first->search(data[query], 10, SearchRequest{}),
                  second->search(data[query], 10, SearchRequest{}));
    }
}

}  // namespace
}  // namespace hopfilter
