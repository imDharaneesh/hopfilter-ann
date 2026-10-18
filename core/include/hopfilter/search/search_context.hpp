#pragma once

#include <cstddef>

#include <hopfilter/common/visited_set.hpp>
#include <hopfilter/graph/hnsw_graph.hpp>
#include <hopfilter/graph/node_distances.hpp>
#include <hopfilter/graph/node_locks.hpp>

namespace hopfilter {

inline constexpr std::size_t kDefaultEfSearch = 64;

/// Per-query search parameters.
struct SearchRequest {
    /// Size of the dynamic candidate list on level 0; larger trades speed for recall.
    std::size_t efSearch = kDefaultEfSearch;
};

/// Everything one traversal needs: index state plus a VisitedSet owned by this
/// traversal alone. When ``locks`` is given, every neighbor list is copied under its
/// node's lock, which makes the traversal safe while other threads insert; without
/// locks the traversal reads the graph directly. Holds references only; all referents
/// must outlive it.
class SearchContext {
public:
    SearchContext(const HnswGraph& graph,
                  const NodeDistances& distances,
                  VisitedSet& visited,
                  const NodeLocks* locks = nullptr) noexcept
        : graph_(&graph), distances_(&distances), visited_(&visited), locks_(locks) {}

    [[nodiscard]] const HnswGraph& graph() const noexcept { return *graph_; }
    [[nodiscard]] const NodeDistances& distances() const noexcept { return *distances_; }
    [[nodiscard]] VisitedSet& visited() const noexcept { return *visited_; }
    [[nodiscard]] const NodeLocks* locks() const noexcept { return locks_; }

private:
    const HnswGraph* graph_;
    const NodeDistances* distances_;
    VisitedSet* visited_;
    const NodeLocks* locks_;
};

}
