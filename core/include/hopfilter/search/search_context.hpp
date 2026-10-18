#pragma once

#include <cstddef>

#include <hopfilter/common/visited_set.hpp>
#include <hopfilter/graph/hnsw_graph.hpp>
#include <hopfilter/graph/node_distances.hpp>

namespace hopfilter {

inline constexpr std::size_t kDefaultEfSearch = 64;

/// Per-query search parameters.
struct SearchRequest {
    /// Size of the dynamic candidate list on level 0; larger trades speed for recall.
    std::size_t efSearch = kDefaultEfSearch;
};

/// Everything one traversal needs: read-only index state plus a VisitedSet owned by
/// this traversal alone. Holds references only; all referents must outlive it.
class SearchContext {
public:
    SearchContext(const HnswGraph& graph,
                  const NodeDistances& distances,
                  VisitedSet& visited) noexcept
        : graph_(&graph), distances_(&distances), visited_(&visited) {}

    [[nodiscard]] const HnswGraph& graph() const noexcept { return *graph_; }
    [[nodiscard]] const NodeDistances& distances() const noexcept { return *distances_; }
    [[nodiscard]] VisitedSet& visited() const noexcept { return *visited_; }

private:
    const HnswGraph* graph_;
    const NodeDistances* distances_;
    VisitedSet* visited_;
};

}  // namespace hopfilter
