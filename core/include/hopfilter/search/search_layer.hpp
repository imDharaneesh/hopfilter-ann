#pragma once

#include <cstddef>
#include <optional>
#include <span>

#include <hopfilter/common/types.hpp>
#include <hopfilter/filter/filter_predicate.hpp>
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

[[nodiscard]] std::optional<Neighbor> descendToLevelZero(const SearchContext& context,
                                                         std::span<const float> query);

struct EfSchedule {
    std::size_t ef;
    std::size_t rejectionWindow;
    double rejectionThreshold;
    double growthFactor;
    std::size_t maxEf;
};

[[nodiscard]] SearchResults filteredSearchLayer(const SearchContext& context,
                                                std::span<const float> query,
                                                std::span<const Neighbor> entryPoints,
                                                const FilterPredicate& filter,
                                                const EfSchedule& schedule);

}
