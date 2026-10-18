#pragma once

#include <cstddef>
#include <span>

#include <hopfilter/common/types.hpp>
#include <hopfilter/search/search_context.hpp>

namespace hopfilter {

/// Strategy for answering a k-nearest-neighbor query against an HNSW graph.
///
/// Implementations are stateless, so one instance serves every thread; per-query
/// state lives in the SearchContext. Results are closest first, without duplicates,
/// and hold at most ``k`` neighbors.
class SearchStrategy {
public:
    virtual ~SearchStrategy() = default;

    [[nodiscard]] virtual SearchResults search(const SearchContext& context,
                                               std::span<const float> query,
                                               std::size_t k,
                                               const SearchRequest& request) const = 0;
};

/// Standard HNSW query: greedy descent to level 1, then a best-first search on level 0
/// with ``max(efSearch, k)`` candidates.
class UnfilteredStrategy final : public SearchStrategy {
public:
    [[nodiscard]] SearchResults search(const SearchContext& context,
                                       std::span<const float> query,
                                       std::size_t k,
                                       const SearchRequest& request) const override;
};

}
