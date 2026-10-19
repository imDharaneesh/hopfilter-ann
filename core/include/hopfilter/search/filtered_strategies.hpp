#pragma once

#include <cstddef>
#include <span>

#include <hopfilter/common/types.hpp>
#include <hopfilter/search/filtered_search_config.hpp>
#include <hopfilter/search/search_context.hpp>
#include <hopfilter/search/search_layer.hpp>
#include <hopfilter/search/search_strategy.hpp>

namespace hopfilter {

class PostFilterStrategy final : public SearchStrategy {
public:
    explicit PostFilterStrategy(double overFetchFactor);

    [[nodiscard]] SearchResults search(const SearchContext& context,
                                       std::span<const float> query,
                                       std::size_t k,
                                       const SearchRequest& request) const override;

private:
    double overFetchFactor_;
};

class PreFilterStrategy final : public SearchStrategy {
public:
    [[nodiscard]] SearchResults search(const SearchContext& context,
                                       std::span<const float> query,
                                       std::size_t k,
                                       const SearchRequest& request) const override;
};

class InGraphStrategy final : public SearchStrategy {
public:
    explicit InGraphStrategy(const FilteredSearchConfig& config);

    [[nodiscard]] SearchResults search(const SearchContext& context,
                                       std::span<const float> query,
                                       std::size_t k,
                                       const SearchRequest& request) const override;

private:
    [[nodiscard]] EfSchedule scheduleFor(std::size_t ef) const noexcept;

    std::size_t rejectionWindow_;
    double rejectionThreshold_;
    double efGrowthFactor_;
    std::size_t maxEf_;
};

}
