#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/filter/attribute_store.hpp>
#include <hopfilter/filter/filter_expression.hpp>
#include <hopfilter/filter/selectivity.hpp>
#include <hopfilter/search/adaptive_strategy.hpp>
#include <hopfilter/search/search_context.hpp>
#include <hopfilter/search/search_strategy.hpp>

#include "support/brute_force.hpp"
#include "support/filtered_index.hpp"

namespace hopfilter {
namespace {

constexpr std::size_t kNeighbors = 10;

class RecordingStrategy final : public SearchStrategy {
public:
    explicit RecordingStrategy(NodeId marker) : marker_(marker) {}

    [[nodiscard]] SearchResults search(
        [[maybe_unused]] const SearchContext& context,
        [[maybe_unused]] std::span<const float> query,
        [[maybe_unused]] std::size_t k,
        [[maybe_unused]] const SearchRequest& request) const override {
        return {Neighbor{marker_, 0.0F}};
    }

private:
    NodeId marker_;
};

class HnswFilteredSearchTest : public ::testing::Test {
public:
    static void SetUpTestSuite() { shared = std::make_unique<testing::FilteredIndex>(47); }
    static void TearDownTestSuite() { shared.reset(); }

    static inline std::unique_ptr<testing::FilteredIndex> shared;
};

TEST_F(HnswFilteredSearchTest, DefaultStrategyAppliesTheFilter) {
    const auto filter =
        (FilterExpression::equals("bucket", 3) | FilterExpression::equals("bucket", 4) |
         FilterExpression::equals("bucket", 5) | FilterExpression::equals("bucket", 6))
            .compile(shared->index().attributes());
    double total = 0.0;
    for (const auto& query : shared->queries()) {
        const auto found =
            shared->index().search(query, kNeighbors, SearchRequest{64, filter.get()});
        EXPECT_TRUE(testing::isWellFormed(found, *filter));
        total += testing::recall(found, shared->truth(query, kNeighbors, *filter));
    }

    EXPECT_GE(total / static_cast<double>(shared->queries().size()), 0.95);
}

TEST_F(HnswFilteredSearchTest, AdaptiveDelegatesToThePlannedStrategy) {
    const AdaptiveStrategy strategy{AdaptiveParts{
        std::make_unique<ExactSelectivityEstimator>(),
        std::make_unique<RecordingStrategy>(1),
        std::make_unique<RecordingStrategy>(2),
        std::make_unique<RecordingStrategy>(3),
        std::make_unique<RecordingStrategy>(4),
        AdaptiveThresholds{0.5, 0.01},
    }};
    const auto& attributes = shared->index().attributes();
    const auto markerFor = [&](const FilterPredicate* filter) {
        return shared->index()
            .search(shared->queries().front(), kNeighbors, SearchRequest{64, filter}, strategy)
            .front()
            .id;
    };
    const auto broad = (!FilterExpression::equals("bucket", 0)).compile(attributes);
    const auto narrow = FilterExpression::equals("bucket", 0).compile(attributes);
    std::vector<FilterExpression> tenth;
    tenth.reserve(10);
    for (CategoryCode bucket = 0; bucket < 10; ++bucket) {
        tenth.push_back(FilterExpression::equals("bucket", bucket));
    }
    const auto middle = FilterExpression::anyOf(tenth).compile(attributes);

    EXPECT_EQ(markerFor(nullptr), 1U);
    EXPECT_EQ(markerFor(broad.get()), 2U);
    EXPECT_EQ(markerFor(middle.get()), 3U);
    EXPECT_EQ(markerFor(narrow.get()), 4U);
}

TEST_F(HnswFilteredSearchTest, RejectsColumnsLongerThanTheCapacity) {
    HnswIndex& index = shared->index();
    const std::size_t tooLong = index.capacity() + 1;

    EXPECT_THROW(index.setCategoricalAttribute("big", CategoricalColumn(tooLong)),
                 InvalidArgumentError);
    EXPECT_THROW(index.setNumericAttribute("big", NumericColumn(tooLong)), InvalidArgumentError);
    EXPECT_THROW(static_cast<void>(index.attributes().numeric("big")), InvalidArgumentError);
}

}
}
