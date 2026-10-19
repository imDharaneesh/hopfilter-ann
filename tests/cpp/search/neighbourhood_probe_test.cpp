#include <algorithm>
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
#include <hopfilter/search/adaptive_strategy.hpp>
#include <hopfilter/search/filtered_search_config.hpp>
#include <hopfilter/search/neighbourhood_probe.hpp>
#include <hopfilter/search/search_context.hpp>
#include <hopfilter/search/search_layer.hpp>

#include "support/filtered_index.hpp"

namespace hopfilter {
namespace {

constexpr std::size_t kSmallProbe = 64;
constexpr double kCornerShare = 0.1;

std::vector<float> corner(float value) {
    return std::vector<float>(testing::kFilteredDimension, value);
}

class NeighbourhoodProbeTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        shared = std::make_unique<testing::FilteredIndex>(53);
        const auto anchor = corner(0.9F);
        NumericColumn distances;
        for (const auto& point : shared->data()) {
            distances.push_back(shared->index().metric().distance(anchor, point));
        }
        auto sorted = distances;
        const auto cut =
            static_cast<std::ptrdiff_t>(kCornerShare * static_cast<double>(sorted.size()));
        std::ranges::nth_element(sorted, sorted.begin() + cut);
        cornerCutoff = sorted[static_cast<std::size_t>(cut)];
        shared->index().setNumericAttribute("corner", std::move(distances));
    }

    static void TearDownTestSuite() { shared.reset(); }

    NeighbourhoodProbeTest()
        : distances_(shared->index().metric(), shared->index().vectors()),
          visited_(shared->index().capacity()),
          context_(shared->index().graph(), distances_, visited_) {}

    [[nodiscard]] const SearchContext& context() const { return context_; }

    [[nodiscard]] static FilterExpression nearCorner() {
        return FilterExpression::between("corner", 0.0, cornerCutoff);
    }

    [[nodiscard]] static PredicatePtr compiled(const FilterExpression& expression) {
        return expression.compile(shared->index().attributes());
    }

    [[nodiscard]] NodeId startFor(const std::vector<float>& query) const {
        const auto start = descendToLevelZero(context_, query);
        return start ? start->id : kInvalidNodeId;
    }

    static inline std::unique_ptr<testing::FilteredIndex> shared;
    static inline double cornerCutoff = 0.0;

private:
    NodeDistances distances_;
    VisitedSet visited_;
    SearchContext context_;
};

TEST_F(NeighbourhoodProbeTest, ShareFollowsTheFilter) {
    const auto everything = compiled(!FilterExpression::equals("bucket", testing::kBuckets));
    const auto nothing = compiled(FilterExpression::equals("bucket", testing::kBuckets));
    const NodeId start = startFor(shared->queries().front());

    EXPECT_DOUBLE_EQ(neighbourhoodShare(context(), start, *everything, kSmallProbe), 1.0);
    EXPECT_DOUBLE_EQ(neighbourhoodShare(context(), start, *nothing, kSmallProbe), 0.0);
}

TEST_F(NeighbourhoodProbeTest, ShareSeparatesNearAndDistantMatches) {
    const auto filter = compiled(nearCorner());

    const double near = neighbourhoodShare(context(), startFor(corner(0.9F)), *filter, kSmallProbe);
    const double distant =
        neighbourhoodShare(context(), startFor(corner(-0.9F)), *filter, kSmallProbe);

    EXPECT_GT(near, 0.5);
    EXPECT_LT(distant, kCornerShare / 4);
}

FilteredSearchConfig smallProbeConfig() {
    FilteredSearchConfig config;
    config.localProbe = kSmallProbe;
    return config;
}

TEST_F(NeighbourhoodProbeTest, AdaptiveSendsDistantQueriesToExactSearch) {
    const auto strategy = AdaptiveStrategy::withDefaults(smallProbeConfig());
    const auto filter = compiled(nearCorner());
    const SearchRequest request{64, filter.get()};

    EXPECT_EQ(strategy->plan(request, shared->index().size()), FilterPlan::InGraph);
    EXPECT_EQ(strategy->planFor(context(), corner(0.9F), request), FilterPlan::InGraph);
    EXPECT_EQ(strategy->planFor(context(), corner(-0.9F), request), FilterPlan::Exact);
}

TEST_F(NeighbourhoodProbeTest, DistantCheckCanBeTurnedOffOrBounded) {
    const auto filter = compiled(nearCorner());
    const SearchRequest request{64, filter.get()};
    auto disabled = smallProbeConfig();
    disabled.thresholds.distantShare = 0.0;
    auto bounded = smallProbeConfig();
    bounded.thresholds.distantExactBelow = kCornerShare / 2;

    EXPECT_EQ(AdaptiveStrategy::withDefaults(disabled)->planFor(context(), corner(-0.9F), request),
              FilterPlan::InGraph);
    EXPECT_EQ(AdaptiveStrategy::withDefaults(bounded)->planFor(context(), corner(-0.9F), request),
              FilterPlan::InGraph);
}

TEST_F(NeighbourhoodProbeTest, DistantQueriesStillGetExactResults) {
    const auto strategy = AdaptiveStrategy::withDefaults(smallProbeConfig());
    const auto filter = compiled(nearCorner());
    const auto query = corner(-0.9F);

    const auto found =
        shared->index().search(query, 10, SearchRequest{64, filter.get()}, *strategy);

    EXPECT_TRUE(testing::isWellFormed(found, *filter));
    EXPECT_DOUBLE_EQ(testing::recall(found, shared->truth(query, 10, *filter)), 1.0);
}

TEST(LocalProbeConfigTest, RejectsInvalidSettings) {
    FilteredSearchConfig config;
    config.localProbe = 0;
    EXPECT_THROW(validate(config), InvalidArgumentError);

    config = FilteredSearchConfig{};
    config.thresholds.distantShare = -0.1;
    EXPECT_THROW(validate(config), InvalidArgumentError);

    config = FilteredSearchConfig{};
    config.thresholds.distantExactBelow = 1.5;
    EXPECT_THROW(validate(config), InvalidArgumentError);
}

}
}
