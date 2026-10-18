#pragma once

#include <vector>

#include <hopfilter/common/types.hpp>
#include <hopfilter/graph/hnsw_graph.hpp>

namespace hopfilter::testing {

/// Pushes every not-yet-reached level-0 neighbor of ``node`` onto ``frontier``.
inline void visitNeighbors(const HnswGraph& graph,
                           NodeId node,
                           std::vector<bool>& reached,
                           std::vector<NodeId>& frontier) {
    for (const NodeId neighbor : graph.neighbors(node, 0)) {
        if (!reached[neighbor]) {
            reached[neighbor] = true;
            frontier.push_back(neighbor);
        }
    }
}

/// Which nodes a traversal from the entry point can reach by following level-0 edges.
inline std::vector<bool> reachableOnLevelZero(const HnswGraph& graph) {
    std::vector<bool> reached(graph.size(), false);
    if (!graph.entryPoint()) {
        return reached;
    }
    std::vector<NodeId> frontier{*graph.entryPoint()};
    reached[frontier.front()] = true;
    while (!frontier.empty()) {
        const NodeId current = frontier.back();
        frontier.pop_back();
        visitNeighbors(graph, current, reached, frontier);
    }
    return reached;
}

}  // namespace hopfilter::testing
