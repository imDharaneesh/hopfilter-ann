#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/filter/filter_predicate.hpp>
#include <hopfilter/filter/selectivity.hpp>
#include <hopfilter/search/adaptive_strategy.hpp>
#include <hopfilter/search/filtered_search_config.hpp>
#include <hopfilter/search/filtered_strategies.hpp>
#include <hopfilter/search/neighbourhood_probe.hpp>
#include <hopfilter/search/search_context.hpp>
#include <hopfilter/search/search_layer.hpp>
#include <hopfilter/search/search_strategy.hpp>
#include <hopfilter/search/strategy_registry.hpp>

namespace hopfilter {

std::string_view filterPlanName(FilterPlan plan) noexcept {
    switch (plan) {
        case FilterPlan::Unfiltered:
            return kUnfilteredStrategyName;
        case FilterPlan::PostFilter:
            return kPostFilterStrategyName;
        case FilterPlan::InGraph:
            return kInGraphStrategyName;
        case FilterPlan::Exact:
            return kPreFilterStrategyName;
    }
    return "unknown";
}

AdaptiveStrategy::AdaptiveStrategy(AdaptiveParts parts)
    : estimator_(std::move(parts.estimator)),
      unfiltered_(std::move(parts.unfiltered)),
      postFilter_(std::move(parts.postFilter)),
      inGraph_(std::move(parts.inGraph)),
      exact_(std::move(parts.exact)),
      thresholds_(parts.thresholds),
      localProbe_(parts.localProbe) {
    if (!estimator_ || !unfiltered_ || !postFilter_ || !inGraph_ || !exact_) {
        throw InvalidArgumentError("adaptive strategy needs an estimator and every delegate");
    }
}

std::unique_ptr<AdaptiveStrategy> AdaptiveStrategy::withDefaults(
    const FilteredSearchConfig& config) {
    validate(config);
    return std::make_unique<AdaptiveStrategy>(AdaptiveParts{
        std::make_unique<SamplingSelectivityEstimator>(config.selectivitySample,
                                                       config.selectivitySeed),
        std::make_unique<UnfilteredStrategy>(),
        std::make_unique<PostFilterStrategy>(config.overFetchFactor),
        std::make_unique<InGraphStrategy>(config),
        std::make_unique<PreFilterStrategy>(),
        config.thresholds,
        config.localProbe,
    });
}

SearchResults AdaptiveStrategy::search(const SearchContext& context,
                                       std::span<const float> query,
                                       std::size_t k,
                                       const SearchRequest& request) const {
    return delegateFor(planFor(context, query, request)).search(context, query, k, request);
}

FilterPlan AdaptiveStrategy::plan(const SearchRequest& request, std::size_t universeSize) const {
    if (request.filter == nullptr) {
        return FilterPlan::Unfiltered;
    }
    const double selectivity = estimator_->estimate(*request.filter, universeSize);
    return planForSelectivity(request, universeSize, selectivity);
}

FilterPlan AdaptiveStrategy::planFor(const SearchContext& context,
                                     std::span<const float> query,
                                     const SearchRequest& request) const {
    if (request.filter == nullptr) {
        return FilterPlan::Unfiltered;
    }
    const std::size_t universeSize = context.graph().size();
    const double selectivity = estimator_->estimate(*request.filter, universeSize);
    const FilterPlan global = planForSelectivity(request, universeSize, selectivity);
    if (global == FilterPlan::InGraph &&
        matchesLookDistant(context, query, *request.filter, selectivity)) {
        return FilterPlan::Exact;
    }
    return global;
}

bool AdaptiveStrategy::matchesLookDistant(const SearchContext& context,
                                          std::span<const float> query,
                                          const FilterPredicate& filter,
                                          double selectivity) const {
    if (thresholds_.distantShare <= 0.0 || selectivity >= thresholds_.distantExactBelow) {
        return false;
    }
    const auto start = descendToLevelZero(context, query);
    if (!start) {
        return false;
    }
    const double localShare = neighbourhoodShare(context, start->id, filter, localProbe_);
    return localShare < thresholds_.distantShare * selectivity;
}

FilterPlan AdaptiveStrategy::planForSelectivity(const SearchRequest& request,
                                                std::size_t universeSize,
                                                double selectivity) const noexcept {
    if (selectivity >= thresholds_.postFilterAbove) {
        return FilterPlan::PostFilter;
    }
    const double expectedMatches = selectivity * static_cast<double>(universeSize);
    if (selectivity < thresholds_.exactBelow ||
        expectedMatches < static_cast<double>(request.efSearch)) {
        return FilterPlan::Exact;
    }
    return FilterPlan::InGraph;
}

const SearchStrategy& AdaptiveStrategy::delegateFor(FilterPlan plan) const noexcept {
    switch (plan) {
        case FilterPlan::PostFilter:
            return *postFilter_;
        case FilterPlan::InGraph:
            return *inGraph_;
        case FilterPlan::Exact:
            return *exact_;
        case FilterPlan::Unfiltered:
            break;
    }
    return *unfiltered_;
}

}
