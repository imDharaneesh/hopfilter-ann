#pragma once

#include <cstddef>

#include <hopfilter/common/visited_set.hpp>
#include <hopfilter/filter/filter_predicate.hpp>
#include <hopfilter/graph/hnsw_graph.hpp>
#include <hopfilter/graph/node_distances.hpp>
#include <hopfilter/graph/node_locks.hpp>

namespace hopfilter {

inline constexpr std::size_t kDefaultEfSearch = 64;

struct SearchRequest {
    std::size_t efSearch = kDefaultEfSearch;
    const FilterPredicate* filter = nullptr;
};

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
