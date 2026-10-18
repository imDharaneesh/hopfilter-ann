#pragma once

#include <cstddef>
#include <span>

#include <hopfilter/common/types.hpp>
#include <hopfilter/search/search_context.hpp>

namespace hopfilter {

/// Best-first search on one level (Algorithm 2 of the HNSW paper).
///
/// Starts from ``entryPoints`` (with their distances to ``query``), keeps the ``ef``
/// closest nodes seen, and stops once the closest unexplored candidate is farther
/// than the farthest kept result. Returns at most ``ef`` neighbors, closest first.
/// Clears the context's VisitedSet before starting.
[[nodiscard]] SearchResults searchLayer(const SearchContext& context,
                                        std::span<const float> query,
                                        std::span<const Neighbor> entryPoints,
                                        std::size_t ef,
                                        LevelIndex level);

/// Greedy (``ef = 1``) descent from ``entry`` through levels ``topLevel`` down to
/// ``stopLevel + 1``. Returns the closest node found; ``entry`` itself if
/// ``topLevel <= stopLevel``.
[[nodiscard]] Neighbor greedyDescend(const SearchContext& context,
                                     std::span<const float> query,
                                     Neighbor entry,
                                     LevelIndex topLevel,
                                     LevelIndex stopLevel);

}
