#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

#include <hopfilter/common/types.hpp>
#include <hopfilter/graph/neighbor_selector.hpp>
#include <hopfilter/graph/node_distances.hpp>

namespace hopfilter {
namespace {

bool closerToBaseThanToSelected(const Neighbor& candidate,
                                const std::vector<NodeId>& selected,
                                const NodeDistances& distances) {
    return std::ranges::all_of(selected, [&](NodeId chosen) {
        return candidate.distance < distances.between(candidate.id, chosen);
    });
}

}  // namespace

std::vector<NodeId> SimpleNeighborSelector::select(std::span<const Neighbor> candidates,
                                                   std::size_t maxNeighbors,
                                                   const NodeDistances& /*distances*/) const {
    const std::size_t count = std::min(maxNeighbors, candidates.size());
    std::vector<NodeId> selected;
    selected.reserve(count);
    for (const Neighbor& candidate : candidates.first(count)) {
        selected.push_back(candidate.id);
    }
    return selected;
}

std::vector<NodeId> HeuristicNeighborSelector::select(std::span<const Neighbor> candidates,
                                                      std::size_t maxNeighbors,
                                                      const NodeDistances& distances) const {
    std::vector<NodeId> selected;
    std::vector<NodeId> pruned;
    selected.reserve(maxNeighbors);
    for (const Neighbor& candidate : candidates) {
        if (selected.size() == maxNeighbors) {
            break;
        }
        if (closerToBaseThanToSelected(candidate, selected, distances)) {
            selected.push_back(candidate.id);
        } else if (keepPrunedConnections_) {
            pruned.push_back(candidate.id);
        }
    }
    const std::size_t refill = std::min(maxNeighbors - selected.size(), pruned.size());
    selected.insert(
        selected.end(), pruned.begin(), pruned.begin() + static_cast<std::ptrdiff_t>(refill));
    return selected;
}

}  // namespace hopfilter
