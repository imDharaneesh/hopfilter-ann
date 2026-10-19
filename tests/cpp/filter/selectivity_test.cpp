#include <cstddef>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/filter/dynamic_bitset.hpp>
#include <hopfilter/filter/filter_predicate.hpp>
#include <hopfilter/filter/selectivity.hpp>

namespace hopfilter {
namespace {

constexpr std::size_t kUniverse = 100'000;

BitsetPredicate everyNth(std::size_t step, std::size_t universe = kUniverse) {
    DynamicBitset bits{universe};
    for (std::size_t id = 0; id < universe; id += step) {
        bits.set(id);
    }
    return BitsetPredicate{bits};
}

TEST(SelectivityTest, ExactEstimateCountsEveryMatch) {
    const ExactSelectivityEstimator estimator;

    EXPECT_DOUBLE_EQ(estimator.estimate(everyNth(4), kUniverse), 0.25);
    EXPECT_DOUBLE_EQ(estimator.estimate(everyNth(4), 0), 0.0);
}

TEST(SelectivityTest, SamplingEstimateIsCloseToTheTruth) {
    const SamplingSelectivityEstimator estimator{4096};

    for (const std::size_t step : {2U, 10U, 100U}) {
        const double truth = 1.0 / static_cast<double>(step);
        EXPECT_NEAR(estimator.estimate(everyNth(step), kUniverse), truth, 0.02) << step;
    }
}

TEST(SelectivityTest, SamplingIsDeterministicForASeed) {
    const SamplingSelectivityEstimator estimator{256, 7};
    const auto predicate = everyNth(3);

    EXPECT_DOUBLE_EQ(estimator.estimate(predicate, kUniverse),
                     estimator.estimate(predicate, kUniverse));
}

TEST(SelectivityTest, SamplingIsExactOnSmallUniverses) {
    const SamplingSelectivityEstimator estimator{1024};

    EXPECT_DOUBLE_EQ(estimator.estimate(everyNth(5, 500), 500), 0.2);
    EXPECT_DOUBLE_EQ(estimator.estimate(everyNth(5, 500), 0), 0.0);
}

TEST(SelectivityTest, SamplingRejectsAnEmptySample) {
    EXPECT_THROW(SamplingSelectivityEstimator{0}, InvalidArgumentError);
}

}
}
