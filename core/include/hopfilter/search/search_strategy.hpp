#pragma once

#include <cstddef>
#include <span>

#include <hopfilter/common/types.hpp>
#include <hopfilter/search/search_context.hpp>

namespace hopfilter {

class SearchStrategy {
public:
    virtual ~SearchStrategy() = default;

    [[nodiscard]] virtual SearchResults search(const SearchContext& context,
                                               std::span<const float> query,
                                               std::size_t k,
                                               const SearchRequest& request) const = 0;
};

class UnfilteredStrategy final : public SearchStrategy {
public:
    [[nodiscard]] SearchResults search(const SearchContext& context,
                                       std::span<const float> query,
                                       std::size_t k,
                                       const SearchRequest& request) const override;
};

}
