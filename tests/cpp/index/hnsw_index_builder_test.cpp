#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/distance/metrics.hpp>
#include <hopfilter/graph/hnsw_graph.hpp>
#include <hopfilter/index/hnsw_index_builder.hpp>
#include <hopfilter/index/index_config.hpp>
#include <hopfilter/storage/vector_store.hpp>

namespace hopfilter {
namespace {

IndexConfig validConfig() {
    IndexConfig config;
    config.dimension = 8;
    config.capacity = 100;
    config.maxNeighbors = 8;
    config.efConstruction = 32;
    return config;
}

void expectRejected(const IndexConfig& config, const std::string& message) {
    try {
        static_cast<void>(HnswIndexBuilder{}.withConfig(config).build());
        FAIL() << "expected InvalidArgumentError: " << message;
    } catch (const InvalidArgumentError& error) {
        EXPECT_EQ(std::string{error.what()}, message);
    }
}

TEST(HnswIndexBuilderTest, BuildsEmptyIndexWithDefaults) {
    const auto index = HnswIndexBuilder{}.withConfig(validConfig()).build();

    EXPECT_EQ(index->size(), 0U);
    EXPECT_EQ(index->dimension(), 8U);
    EXPECT_EQ(index->capacity(), 100U);
    EXPECT_EQ(index->metric().name(), "l2");
    EXPECT_EQ(index->graph().maxNeighbors(0), 16U);
    EXPECT_EQ(index->graph().maxNeighbors(1), 8U);
}

TEST(HnswIndexBuilderTest, CreatesMetricFromConfigName) {
    IndexConfig config = validConfig();
    config.metricName = "cosine";

    EXPECT_EQ(HnswIndexBuilder{}.withConfig(config).build()->metric().name(), "cosine");
}

TEST(HnswIndexBuilderTest, RejectsInvalidConfigurationWithFieldName) {
    auto config = validConfig();
    config.dimension = 0;
    expectRejected(config, "dimension must be positive");

    config = validConfig();
    config.capacity = 0;
    expectRejected(config, "capacity must be positive");

    config = validConfig();
    config.maxNeighbors = 1;
    expectRejected(config, "maxNeighbors must be at least 2");

    config = validConfig();
    config.efConstruction = 4;
    expectRejected(config, "efConstruction must be at least maxNeighbors");
}

TEST(HnswIndexBuilderTest, RejectsUnknownMetric) {
    auto config = validConfig();
    config.metricName = "hamming";

    EXPECT_THROW(static_cast<void>(HnswIndexBuilder{}.withConfig(config).build()),
                 InvalidArgumentError);
}

TEST(HnswIndexBuilderTest, RejectsInjectedMetricThatDisagreesWithConfig) {
    HnswIndexBuilder builder;
    builder.withConfig(validConfig()).withMetric(std::make_unique<CosineMetric>());

    EXPECT_THROW(static_cast<void>(builder.build()), InvalidArgumentError);
}

TEST(HnswIndexBuilderTest, RequiresConfigAndIsEmptyAfterBuild) {
    HnswIndexBuilder builder;
    EXPECT_THROW(static_cast<void>(builder.build()), InvalidArgumentError);

    builder.withConfig(validConfig());
    static_cast<void>(builder.build());

    EXPECT_THROW(static_cast<void>(builder.build()), InvalidArgumentError);
}

TEST(HnswIndexBuilderTest, RejectsRestoredStorageWithWrongShape) {
    const auto config = validConfig();
    HnswIndexBuilder builder;
    builder.withConfig(config).withStorage(std::make_unique<VectorStore>(4, config.capacity),
                                           std::make_unique<HnswGraph>(config.capacity, 16, 8));

    EXPECT_THROW(static_cast<void>(builder.build()), InvalidArgumentError);
}

TEST(HnswIndexBuilderTest, RejectsHalfRestoredStorage) {
    HnswIndexBuilder builder;
    builder.withConfig(validConfig()).withStorage(std::make_unique<VectorStore>(8, 100), nullptr);

    EXPECT_THROW(static_cast<void>(builder.build()), InvalidArgumentError);
}

}
}
