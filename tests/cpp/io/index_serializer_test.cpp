#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/index/hnsw_index_builder.hpp>
#include <hopfilter/index/index_config.hpp>
#include <hopfilter/io/crc32.hpp>
#include <hopfilter/io/index_serializer.hpp>

#include "support/brute_force.hpp"
#include "support/random_vectors.hpp"

namespace hopfilter {
namespace {

constexpr std::size_t kDimension = 8;
constexpr std::size_t kPoints = 300;
constexpr std::size_t kFixturePoints = 64;
constexpr std::uint64_t kFixtureSeed = 99;

IndexConfig configFor(std::string metricName) {
    IndexConfig config;
    config.maxNeighbors = 6;
    config.efConstruction = 24;
    config.metricName = std::move(metricName);
    return config;
}

std::string serialized(const HnswIndex& index) {
    std::ostringstream sink;
    IndexSerializer::save(index, sink);
    return sink.str();
}

std::unique_ptr<HnswIndex> deserialized(const std::string& bytes) {
    std::istringstream source{bytes};
    return IndexSerializer::load(source);
}

void expectSameGraph(const HnswGraph& expected, const HnswGraph& actual) {
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_EQ(actual.entryPoint(), expected.entryPoint());
    EXPECT_EQ(actual.maxLevel(), expected.maxLevel());
    for (NodeId id = 0; id < expected.size(); ++id) {
        ASSERT_EQ(actual.levelOf(id), expected.levelOf(id));
        for (LevelIndex level = 0; level <= expected.levelOf(id); ++level) {
            const auto lhs = expected.neighbors(id, level);
            const auto rhs = actual.neighbors(id, level);
            EXPECT_TRUE(std::ranges::equal(lhs, rhs)) << "node " << id << " level " << int{level};
        }
    }
}

std::string serializationErrorOf(const std::string& bytes) {
    try {
        static_cast<void>(deserialized(bytes));
    } catch (const SerializationError& error) {
        return error.what();
    }
    return "<loaded without error>";
}

TEST(Crc32Test, MatchesStandardCheckValue) {
    constexpr std::string_view kCheckInput = "123456789";
    Crc32 crc;

    crc.update(std::as_bytes(std::span{kCheckInput}));

    EXPECT_EQ(crc.value(), 0xCBF43926U);
}

class RoundTripTest : public ::testing::TestWithParam<std::string> {};

TEST_P(RoundTripTest, PreservesConfigVectorsGraphAndResults) {
    testing::RandomVectors random{7};
    const auto data = random.vectors(kPoints, kDimension);
    const auto original = testing::buildIndex(data, configFor(GetParam()));

    const auto restored = deserialized(serialized(*original));

    EXPECT_EQ(restored->config().metricName, GetParam());
    EXPECT_EQ(restored->config().maxNeighbors, original->config().maxNeighbors);
    EXPECT_EQ(restored->capacity(), original->capacity());
    for (NodeId id = 0; id < kPoints; ++id) {
        ASSERT_TRUE(std::ranges::equal(restored->vectors().at(id), original->vectors().at(id)));
    }
    expectSameGraph(original->graph(), restored->graph());
    for (std::size_t query = 0; query < 20; ++query) {
        EXPECT_EQ(restored->search(data[query], 10, SearchRequest{}),
                  original->search(data[query], 10, SearchRequest{}));
    }
}

INSTANTIATE_TEST_SUITE_P(Metrics, RoundTripTest, ::testing::Values("l2", "ip", "cosine"));

TEST(IndexSerializerTest, EmptyIndexRoundTrips) {
    IndexConfig config = configFor("l2");
    config.dimension = kDimension;
    config.capacity = 4;
    const auto empty = HnswIndexBuilder{}.withConfig(config).build();

    const auto restored = deserialized(serialized(*empty));

    EXPECT_EQ(restored->size(), 0U);
    EXPECT_FALSE(restored->graph().entryPoint().has_value());
}

TEST(IndexSerializerTest, RestoredIndexAcceptsMoreInserts) {
    testing::RandomVectors random{8};
    auto data = random.vectors(50, kDimension);
    IndexConfig config = configFor("l2");
    config.dimension = kDimension;
    config.capacity = 60;
    auto original = HnswIndexBuilder{}.withConfig(config).build();
    for (const auto& point : data) {
        static_cast<void>(original->insert(point));
    }
    const auto restored = deserialized(serialized(*original));

    const auto extra = random.vector(kDimension);
    const NodeId id = restored->insert(extra);

    EXPECT_EQ(id, 50U);
    EXPECT_EQ(restored->search(extra, 1, SearchRequest{}).front().id, 50U);
}

TEST(IndexSerializerTest, EveryTruncationIsRejected) {
    testing::RandomVectors random{9};
    const auto bytes = serialized(*testing::buildIndex(random.vectors(40, 4), configFor("l2")));

    for (std::size_t length = 0; length < bytes.size(); ++length) {
        EXPECT_NE(serializationErrorOf(bytes.substr(0, length)), "<loaded without error>")
            << "prefix of length " << length << " was accepted";
    }
}

TEST(IndexSerializerTest, FlippedByteFailsTheChecksum) {
    testing::RandomVectors random{10};
    auto bytes = serialized(*testing::buildIndex(random.vectors(40, 4), configFor("l2")));
    const std::size_t insideVectors = 80;
    bytes[insideVectors] = static_cast<char>(bytes[insideVectors] ^ 0x01);

    EXPECT_EQ(serializationErrorOf(bytes), "index checksum mismatch; the data is corrupted");
}

TEST(IndexSerializerTest, RejectsForeignDataAndUnknownVersions) {
    testing::RandomVectors random{11};
    auto bytes = serialized(*testing::buildIndex(random.vectors(10, 4), configFor("l2")));
    auto futureVersion = bytes;
    futureVersion[4] = 2;

    EXPECT_EQ(serializationErrorOf("PNG\x89 not an index"),
              "not a hopfilter index (bad magic bytes)");
    EXPECT_EQ(serializationErrorOf(futureVersion), "unsupported index format version 2");
}

TEST(IndexSerializerTest, FileHelpersRoundTripAndReportMissingFiles) {
    testing::RandomVectors random{12};
    const auto data = random.vectors(30, kDimension);
    const auto original = testing::buildIndex(data, configFor("cosine"));
    const auto path = std::filesystem::temp_directory_path() / "hopfilter_serializer_test.hopf";

    IndexSerializer::saveToFile(*original, path);
    const auto restored = IndexSerializer::loadFromFile(path);
    std::filesystem::remove(path);

    EXPECT_EQ(restored->size(), 30U);
    EXPECT_THROW(static_cast<void>(IndexSerializer::loadFromFile(path)), SerializationError);
}

std::unique_ptr<HnswIndex> fixtureIndex() {
    testing::RandomVectors random{kFixtureSeed};
    return testing::buildIndex(random.vectors(kFixturePoints, kDimension), configFor("l2"));
}

std::filesystem::path fixturePath() {
    return std::filesystem::path{HOPFILTER_FIXTURE_DIR} / "index_format_v1.hopf";
}

// Run explicitly to regenerate the fixture:
//   hopfilter_core_tests --gtest_also_run_disabled_tests --gtest_filter=*WriteVersionOneFixture
TEST(IndexFormatRegressionTest, DISABLED_WriteVersionOneFixture) {
    IndexSerializer::saveToFile(*fixtureIndex(), fixturePath());
}

TEST(IndexFormatRegressionTest, VersionOneFixtureStillLoads) {
    const auto expected = fixtureIndex();

    const auto loaded = IndexSerializer::loadFromFile(fixturePath());

    EXPECT_EQ(loaded->size(), kFixturePoints);
    expectSameGraph(expected->graph(), loaded->graph());
}

}  // namespace
}  // namespace hopfilter
