#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/distance/kernels.hpp>

#include "support/random_vectors.hpp"

namespace hopfilter {
namespace {

constexpr std::size_t kMaxDimension = 67;
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

class ActiveKernelSet {
public:
    explicit ActiveKernelSet(KernelSet set) : previous_(activeKernels().set) { useKernelSet(set); }
    ActiveKernelSet(const ActiveKernelSet&) = delete;
    ActiveKernelSet& operator=(const ActiveKernelSet&) = delete;
    ActiveKernelSet(ActiveKernelSet&&) = delete;
    ActiveKernelSet& operator=(ActiveKernelSet&&) = delete;
    ~ActiveKernelSet() { useKernelSet(previous_); }

private:
    KernelSet previous_;
};

std::string kernelSetTestName(const ::testing::TestParamInfo<KernelSet>& param) {
    return std::string{kernelSetName(param.param)};
}

class KernelSetTest : public ::testing::TestWithParam<KernelSet> {
protected:
    void SetUp() override {
        if (!isSupported(GetParam())) {
            GTEST_SKIP() << kernelSetName(GetParam()) << " is not supported here";
        }
    }
};

TEST_P(KernelSetTest, MatchesDoublePrecisionReferenceForEveryTailLength) {
    const ActiveKernelSet active{GetParam()};
    testing::RandomVectors random{11};
    for (std::size_t dimension = 1; dimension <= kMaxDimension; ++dimension) {
        const auto lhs = random.vector(dimension);
        const auto rhs = random.vector(dimension);

        expectClose(squaredL2(lhs, rhs), referenceSquaredL2(lhs, rhs));
        expectClose(dotProduct(lhs, rhs), referenceDot(lhs, rhs));
    }
}

TEST_P(KernelSetTest, HandComputedValues) {
    const ActiveKernelSet active{GetParam()};
    const std::vector<float> lhs{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    const std::vector<float> rhs{2.0F, 0.0F, 3.0F, 1.0F, -1.0F};

    EXPECT_FLOAT_EQ(squaredL2(lhs, rhs), 1.0F + 4.0F + 0.0F + 9.0F + 36.0F);
    EXPECT_FLOAT_EQ(dotProduct(lhs, rhs), 2.0F + 0.0F + 9.0F + 4.0F - 5.0F);
}

TEST_P(KernelSetTest, EmptyVectorsGiveZero) {
    const ActiveKernelSet active{GetParam()};
    const std::vector<float> empty;

    EXPECT_FLOAT_EQ(squaredL2(empty, empty), 0.0F);
    EXPECT_FLOAT_EQ(dotProduct(empty, empty), 0.0F);
}

INSTANTIATE_TEST_SUITE_P(Sets,
                         KernelSetTest,
                         ::testing::Values(KernelSet::Scalar, KernelSet::Avx2),
                         kernelSetTestName);

TEST(KernelDispatchTest, ScalarIsAlwaysSupportedAndBestIsActiveByDefault) {
    EXPECT_TRUE(isSupported(KernelSet::Scalar));
    EXPECT_TRUE(isSupported(bestSupportedKernelSet()));
    EXPECT_EQ(activeKernels().set, bestSupportedKernelSet());
}

TEST(KernelDispatchTest, SwitchingChangesTheActiveSet) {
    {
        const ActiveKernelSet active{KernelSet::Scalar};
        EXPECT_EQ(activeKernels().set, KernelSet::Scalar);
    }
    EXPECT_EQ(activeKernels().set, bestSupportedKernelSet());
}

TEST(KernelDispatchTest, UnsupportedSetIsRejected) {
    if (isSupported(KernelSet::Avx2)) {
        GTEST_SKIP() << "every set is supported on this machine";
    }
    EXPECT_THROW(useKernelSet(KernelSet::Avx2), InvalidArgumentError);
}

TEST(KernelDispatchTest, NamesAreLowerCase) {
    EXPECT_EQ(kernelSetName(KernelSet::Scalar), "scalar");
    EXPECT_EQ(kernelSetName(KernelSet::Avx2), "avx2");
}

}
}
