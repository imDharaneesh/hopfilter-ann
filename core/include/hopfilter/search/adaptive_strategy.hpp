#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include <hopfilter/common/types.hpp>
#include <hopfilter/filter/selectivity.hpp>
#include <hopfilter/search/filtered_search_config.hpp>
#include <hopfilter/search/search_context.hpp>
#include <hopfilter/search/search_strategy.hpp>

namespace hopfilter {

enum class FilterPlan : std::uint8_t { Unfiltered, PostFilter, InGraph, Exact };

[[nodiscard]] std::string_view filterPlanName(FilterPlan plan) noexcept;

struct AdaptiveParts {
    std::unique_ptr<SelectivityEstimator> estimator;
    std::unique_ptr<SearchStrategy> unfiltered;
    std::unique_ptr<SearchStrategy> postFilter;
    std::unique_ptr<SearchStrategy> inGraph;
    std::unique_ptr<SearchStrategy> exact;
    AdaptiveThresholds thresholds;
};

class AdaptiveStrategy final : public SearchStrategy {
public:
    explicit AdaptiveStrategy(AdaptiveParts parts);

    [[nodiscard]] static std::unique_ptr<AdaptiveStrategy> withDefaults(
        const FilteredSearchConfig& config = {});

    [[nodiscard]] SearchResults search(const SearchContext& context,
                                       std::span<const float> query,
                                       std::size_t k,
                                       const SearchRequest& request) const override;

    [[nodiscard]] FilterPlan plan(const SearchRequest& request, std::size_t universeSize) const;

private:
    [[nodiscard]] const SearchStrategy& delegateFor(FilterPlan plan) const noexcept;

    std::unique_ptr<SelectivityEstimator> estimator_;
    std::unique_ptr<SearchStrategy> unfiltered_;
    std::unique_ptr<SearchStrategy> postFilter_;
    std::unique_ptr<SearchStrategy> inGraph_;
    std::unique_ptr<SearchStrategy> exact_;
    AdaptiveThresholds thresholds_;
};

}
