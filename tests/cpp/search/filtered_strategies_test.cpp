#include <cstddef>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/common/visited_set.hpp>
#include <hopfilter/filter/filter_expression.hpp>
#include <hopfilter/filter/filter_predicate.hpp>
#include <hopfilter/graph/node_distances.hpp>
#include <hopfilter/search/filtered_search_config.hpp>
#include <hopfilter/search/filtered_strategies.hpp>
#include <hopfilter/search/search_context.hpp>
#include <hopfilter/search/search_layer.hpp>
#include <hopfilter/search/search_strategy.hpp>

#include "support/brute_force.hpp"
#include "support/filtered_index.hpp"

namespace hopfilter {
namespace {

constexpr std::size_t kNeighbors = 10;

class FilteredStrategiesTest : public ::testing::Test {
public:
    static void SetUpTestSuite() { shared = std::make_unique<testing::FilteredIndex>(31); }
    static void TearDownTestSuite() { shared.reset(); }

    static PredicatePtr bucketsBelow(CategoryCode limit) {
        std::vector<FilterExpression> buckets;
        buckets.reserve(limit);
        for (CategoryCode bucket = 0; bucket < limit; ++bucket) {
            buckets.push_back(FilterExpression::equals("bucket", bucket));
        }
        return FilterExpression::anyOf(buckets).compile(shared->index().attributes());
    }

    static double meanRecall(const SearchStrategy& strategy,
                             const FilterPredicate& filter,
                             std::size_t ef) {
        double total = 0.0;
        for (const auto& query : shared->queries()) {
            const auto found =
                shared->index().search(query, kNeighbors, SearchRequest{ef, &filter}, strategy);
            EXPECT_TRUE(testing::isWellFormed(found, filter));
            total += testing::recall(found, shared->truth(query, kNeighbors, filter));
        }
        return total / static_cast<double>(shared->queries().size());
    }

    static inline std::unique_ptr<testing::FilteredIndex> shared;
};

TEST_F(FilteredStrategiesTest, PreFilterIsExact) {
    const PreFilterStrategy strategy;
    for (const CategoryCode limit : {1U, 10U, 100U}) {
        const auto filter = bucketsBelow(limit);
        EXPECT_DOUBLE_EQ(meanRecall(strategy, *filter, kNeighbors), 1.0) << limit;
    }
}

TEST_F(FilteredStrategiesTest, InGraphKeepsRecallUnderSelectiveFilters) {
    const InGraphStrategy strategy{FilteredSearchConfig{}};
    for (const CategoryCode limit : {5U, 10U, 50U}) {
        const auto filter = bucketsBelow(limit);
        EXPECT_GE(meanRecall(strategy, *filter, 64), 0.95) << limit;
    }
}

TEST_F(FilteredStrategiesTest, PostFilterMatchesUnfilteredWhenEverythingPasses) {
    const PostFilterStrategy postFilter{4.0};
    const UnfilteredStrategy unfiltered;
    const auto everything = bucketsBelow(testing::kBuckets);
    for (const auto& query : shared->queries()) {
        EXPECT_EQ(shared->index().search(
                      query, kNeighbors, SearchRequest{64, everything.get()}, postFilter),
                  shared->index().search(query, kNeighbors, SearchRequest{64}, unfiltered));
    }
}

TEST_F(FilteredStrategiesTest, PostFilterFallsShortUnderStrictFilters) {
    const PostFilterStrategy strategy{4.0};
    const auto filter = bucketsBelow(1);
    std::size_t returned = 0;
    for (const auto& query : shared->queries()) {
        const auto found =
            shared->index().search(query, kNeighbors, SearchRequest{64, filter.get()}, strategy);
        EXPECT_TRUE(testing::isWellFormed(found, *filter));
        returned += found.size();
    }

    EXPECT_LT(returned, kNeighbors * shared->queries().size() / 2);
}

TEST_F(FilteredStrategiesTest, FilteredStrategiesRequireAFilter) {
    const auto& query = shared->queries().front();
    const PostFilterStrategy postFilter{4.0};
    const PreFilterStrategy preFilter;
    const InGraphStrategy inGraph{FilteredSearchConfig{}};

    for (const SearchStrategy* strategy :
         std::vector<const SearchStrategy*>{&postFilter, &preFilter, &inGraph}) {
        EXPECT_THROW(
            static_cast<void>(shared->index().search(query, 5, SearchRequest{}, *strategy)),
            InvalidArgumentError);
    }
}

TEST_F(FilteredStrategiesTest, UnfilteredStrategyRejectsAFilter) {
    const UnfilteredStrategy strategy;
    const auto filter = bucketsBelow(10);

    EXPECT_THROW(static_cast<void>(shared->index().search(
                     shared->queries().front(), 5, SearchRequest{64, filter.get()}, strategy)),
                 InvalidArgumentError);
}

TEST_F(FilteredStrategiesTest, FilterWithoutMatchesReturnsNothing) {
    const auto none = FilterExpression::equals("bucket", testing::kBuckets + 1)
                          .compile(shared->index().attributes());
    const PreFilterStrategy preFilter;
    const InGraphStrategy inGraph{FilteredSearchConfig{}};

    for (const SearchStrategy* strategy :
         std::vector<const SearchStrategy*>{&preFilter, &inGraph}) {
        EXPECT_TRUE(
            shared->index()
                .search(shared->queries().front(), 5, SearchRequest{64, none.get()}, *strategy)
                .empty());
    }
}

TEST_F(FilteredStrategiesTest, RejectionWidensTheResultSet) {
    const HnswIndex& index = shared->index();
    const NodeDistances distances{index.metric(), index.vectors()};
    VisitedSet visited{index.capacity()};
    const SearchContext context{index.graph(), distances, visited};
    const auto filter = bucketsBelow(20);
    const auto& query = shared->queries().front();
    const auto start = descendToLevelZero(context, query);
    if (!start) {
        FAIL() << "the index has no entry point";
    }

    const EfSchedule fixed{8, 16, 0.5, 2.0, 8};
    const EfSchedule widening{8, 16, 0.5, 2.0, 128};
    const auto narrow = filteredSearchLayer(context, query, {&*start, 1}, *filter, fixed);
    const auto wide = filteredSearchLayer(context, query, {&*start, 1}, *filter, widening);

    EXPECT_EQ(narrow.size(), 8U);
    EXPECT_GT(wide.size(), narrow.size());
    EXPECT_TRUE(testing::isWellFormed(wide, *filter));
}

TEST(FilteredSearchConfigTest, RejectsInconsistentSettings) {
    FilteredSearchConfig config;
    config.thresholds.exactBelow = 0.6;
    config.thresholds.postFilterAbove = 0.5;
    EXPECT_THROW(validate(config), InvalidArgumentError);

    config = FilteredSearchConfig{};
    config.efGrowthFactor = 0.5;
    EXPECT_THROW(validate(config), InvalidArgumentError);

    config = FilteredSearchConfig{};
    config.rejectionWindow = 0;
    EXPECT_THROW(InGraphStrategy{config}, InvalidArgumentError);
    EXPECT_THROW(PostFilterStrategy{0.5}, InvalidArgumentError);
}

}
}
