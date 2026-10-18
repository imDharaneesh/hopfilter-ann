#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/distance/kernels.hpp>

#include "support/random_vectors.hpp"

namespace hopfilter {
namespace {

constexpr std::size_t kMaxDimension = 37;
constexpr float kRelativeTolerance = 1e-5F;

double referenceSquaredL2(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    double total = 0.0;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        const double difference = static_cast<double>(lhs[index]) - rhs[index];
        total += difference * difference;
    }
    return total;
}

double referenceDot(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    double total = 0.0;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        total += static_cast<double>(lhs[index]) * rhs[index];
    }
    return total;
}

void expectClose(float actual, double expected) {
    const auto tolerance = static_cast<double>(kRelativeTolerance) * (1.0 + std::abs(expected));
    EXPECT_NEAR(actual, expected, tolerance);
}

TEST(KernelsTest, MatchDoublePrecisionReferenceForEveryTailLength) {
    testing::RandomVectors random{11};
    for (std::size_t dimension = 1; dimension <= kMaxDimension; ++dimension) {
        const auto lhs = random.vector(dimension);
        const auto rhs = random.vector(dimension);

        expectClose(squaredL2(lhs, rhs), referenceSquaredL2(lhs, rhs));
        expectClose(dotProduct(lhs, rhs), referenceDot(lhs, rhs));
    }
}

TEST(KernelsTest, HandComputedValues) {
    const std::vector<float> lhs{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    const std::vector<float> rhs{2.0F, 0.0F, 3.0F, 1.0F, -1.0F};

    EXPECT_FLOAT_EQ(squaredL2(lhs, rhs), 1.0F + 4.0F + 0.0F + 9.0F + 36.0F);
    EXPECT_FLOAT_EQ(dotProduct(lhs, rhs), 2.0F + 0.0F + 9.0F + 4.0F - 5.0F);
}

TEST(KernelsTest, EmptyVectorsGiveZero) {
    const std::vector<float> empty;

    EXPECT_FLOAT_EQ(squaredL2(empty, empty), 0.0F);
    EXPECT_FLOAT_EQ(dotProduct(empty, empty), 0.0F);
}

}  // namespace
}  // namespace hopfilter
