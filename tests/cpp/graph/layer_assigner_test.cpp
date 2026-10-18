#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/graph/layer_assigner.hpp>

namespace hopfilter {
namespace {

constexpr std::size_t kDraws = 200'000;

std::vector<LevelIndex> drawLevels(std::size_t maxNeighbors, std::uint64_t seed, std::size_t n) {
    ExponentialLayerAssigner assigner{maxNeighbors, seed};
    std::vector<LevelIndex> levels(n);
    for (auto& level : levels) {
        level = assigner.nextLevel();
    }
    return levels;
}

double fractionAtLeast(const std::vector<LevelIndex>& levels, LevelIndex minimum) {
    const auto count =
        std::ranges::count_if(levels, [minimum](LevelIndex level) { return level >= minimum; });
    return static_cast<double>(count) / static_cast<double>(levels.size());
}

TEST(ExponentialLayerAssignerTest, SameSeedGivesSameLevels) {
    EXPECT_EQ(drawLevels(16, 7, 1000), drawLevels(16, 7, 1000));
    EXPECT_NE(drawLevels(16, 7, 1000), drawLevels(16, 8, 1000));
}

TEST(ExponentialLayerAssignerTest, LevelMultiplierIsInverseLogOfM) {
    const ExponentialLayerAssigner assigner{16, 0};

    EXPECT_DOUBLE_EQ(assigner.levelMultiplier(), 1.0 / std::log(16.0));
}

TEST(ExponentialLayerAssignerTest, ReachesLevelLWithProbabilityMToTheMinusL) {
    const auto levels = drawLevels(16, 42, kDraws);

    EXPECT_NEAR(fractionAtLeast(levels, 1), 1.0 / 16.0, 0.003);
    EXPECT_NEAR(fractionAtLeast(levels, 2), 1.0 / 256.0, 0.0006);
    EXPECT_GT(fractionAtLeast(levels, 0), 0.999);
}

TEST(ExponentialLayerAssignerTest, NeverExceedsMaxLevel) {
    const auto levels = drawLevels(2, 3, kDraws);

    EXPECT_LE(std::ranges::max(levels), ExponentialLayerAssigner::kMaxLevel);
}

TEST(ExponentialLayerAssignerTest, RejectsFewerThanTwoNeighbors) {
    EXPECT_THROW(ExponentialLayerAssigner(1, 0), InvalidArgumentError);
    EXPECT_THROW(ExponentialLayerAssigner(0, 0), InvalidArgumentError);
}

}  // namespace
}  // namespace hopfilter
