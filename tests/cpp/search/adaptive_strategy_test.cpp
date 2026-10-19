#include <cstddef>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/filter/dynamic_bitset.hpp>
#include <hopfilter/filter/filter_predicate.hpp>
#include <hopfilter/filter/selectivity.hpp>
#include <hopfilter/search/adaptive_strategy.hpp>
#include <hopfilter/search/filtered_search_config.hpp>
#include <hopfilter/search/search_context.hpp>
#include <hopfilter/search/search_strategy.hpp>

namespace hopfilter {
namespace {

constexpr std::size_t kUniverse = 100'000;

BitsetPredicate fraction(double share) {
    DynamicBitset bits{kUniverse};
    const auto matching = static_cast<std::size_t>(share * static_cast<double>(kUniverse));
    for (std::size_t id = 0; id < matching; ++id) {
        bits.set(id);
    }
    return BitsetPredicate{bits};
}

TEST(AdaptiveStrategyTest, PlansBySelectivity) {
    const auto strategy = AdaptiveStrategy::withDefaults();
    const auto plan = [&](double share) {
        const auto filter = fraction(share);
        return strategy->plan(SearchRequest{64, &filter}, kUniverse);
    };

    EXPECT_EQ(strategy->plan(SearchRequest{}, kUniverse), FilterPlan::Unfiltered);
    EXPECT_EQ(plan(1.0), FilterPlan::PostFilter);
    EXPECT_EQ(plan(0.95), FilterPlan::PostFilter);
    EXPECT_EQ(plan(0.6), FilterPlan::InGraph);
    EXPECT_EQ(plan(0.1), FilterPlan::InGraph);
    EXPECT_EQ(plan(0.01), FilterPlan::Exact);
    EXPECT_EQ(plan(0.001), FilterPlan::Exact);
}

TEST(AdaptiveStrategyTest, UsesExactSearchWhenFewerMatchesThanEf) {
    const AdaptiveStrategy strategy{AdaptiveParts{
        std::make_unique<ExactSelectivityEstimator>(),
        std::make_unique<UnfilteredStrategy>(),
        std::make_unique<UnfilteredStrategy>(),
        std::make_unique<UnfilteredStrategy>(),
        std::make_unique<UnfilteredStrategy>(),
        AdaptiveThresholds{0.5, 0.0},
    }};
    const auto filter = fraction(0.0005);

    EXPECT_EQ(strategy.plan(SearchRequest{64, &filter}, kUniverse), FilterPlan::Exact);
    EXPECT_EQ(strategy.plan(SearchRequest{16, &filter}, kUniverse), FilterPlan::InGraph);
}

TEST(AdaptiveStrategyTest, NamesPlansAfterTheirStrategies) {
    EXPECT_EQ(filterPlanName(FilterPlan::Unfiltered), "unfiltered");
    EXPECT_EQ(filterPlanName(FilterPlan::PostFilter), "post_filter");
    EXPECT_EQ(filterPlanName(FilterPlan::InGraph), "in_graph");
    EXPECT_EQ(filterPlanName(FilterPlan::Exact), "pre_filter");
}

TEST(AdaptiveStrategyTest, RejectsMissingDelegates) {
    EXPECT_THROW(AdaptiveStrategy{AdaptiveParts{}}, InvalidArgumentError);
}

TEST(AdaptiveStrategyTest, RejectsInvalidConfig) {
    FilteredSearchConfig config;
    config.selectivitySample = 0;

    EXPECT_THROW(static_cast<void>(AdaptiveStrategy::withDefaults(config)), InvalidArgumentError);
}

}
}
