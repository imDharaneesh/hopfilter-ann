#pragma once

#include <cstddef>
#include <span>

#include <hopfilter/common/types.hpp>
#include <hopfilter/search/search_context.hpp>

namespace hopfilter {

class Insertable {
public:
    virtual ~Insertable() = default;

    virtual NodeId insert(std::span<const float> vector) = 0;
};

class Searchable {
public:
    virtual ~Searchable() = default;

    [[nodiscard]] virtual SearchResults search(std::span<const float> query,
                                               std::size_t k,
                                               const SearchRequest& request) const = 0;

    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    [[nodiscard]] virtual Dimension dimension() const noexcept = 0;
};

}
