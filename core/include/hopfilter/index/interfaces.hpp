#pragma once

#include <cstddef>
#include <span>

#include <hopfilter/common/types.hpp>
#include <hopfilter/search/search_context.hpp>

namespace hopfilter {

/// Something vectors can be added to.
class Insertable {
public:
    virtual ~Insertable() = default;

    /// Adds ``vector`` and returns its id.
    virtual NodeId insert(std::span<const float> vector) = 0;
};

/// Something that answers k-nearest-neighbor queries. Benchmark adapters for other
/// libraries implement only this interface.
class Searchable {
public:
    virtual ~Searchable() = default;

    /// Up to ``k`` neighbors of ``query``, closest first.
    [[nodiscard]] virtual SearchResults search(std::span<const float> query,
                                               std::size_t k,
                                               const SearchRequest& request) const = 0;

    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    [[nodiscard]] virtual Dimension dimension() const noexcept = 0;
};

}
