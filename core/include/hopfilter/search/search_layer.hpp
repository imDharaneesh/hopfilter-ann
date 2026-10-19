#pragma once

#include <cstddef>
#include <span>

#include <hopfilter/common/types.hpp>
#include <hopfilter/search/search_context.hpp>

namespace hopfilter {

[[nodiscard]] SearchResults searchLayer(const SearchContext& context,
                                        std::span<const float> query,
                                        std::span<const Neighbor> entryPoints,
                                        std::size_t ef,
                                        LevelIndex level);

[[nodiscard]] Neighbor greedyDescend(const SearchContext& context,
                                     std::span<const float> query,
                                     Neighbor entry,
                                     LevelIndex topLevel,
                                     LevelIndex stopLevel);

}
