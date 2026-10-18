#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <hopfilter/common/types.hpp>
#include <hopfilter/graph/neighbor_selector.hpp>
#include <hopfilter/graph/node_distances.hpp>

namespace hopfilter {
namespace {

enum class Verdict : std::uint8_t { Keep, Prune, Duplicate };

struct Judgement {
    Verdict verdict;
    std::size_t duplicateOf;
};

Judgement judge(const Neighbor& candidate,
                const std::vector<NodeId>& selected,
                const NodeDistances& distances) {
    for (std::size_t slot = 0; slot < selected.size(); ++slot) {
        if (candidate.distance > distances.between(candidate.id, selected[slot])) {
            return {Verdict::Prune, slot};
        }
        if (distances.identical(candidate.id, selected[slot])) {
            return {Verdict::Duplicate, slot};
        }
    }
    return {Verdict::Keep, selected.size()};
}

}

std::vector<NodeId> SimpleNeighborSelector::select(
    std::span<const Neighbor> candidates,
    std::size_t maxNeighbors,
    [[maybe_unused]] const NodeDistances& distances) const {
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
        const auto [verdict, slot] = judge(candidate, selected, distances);
        if (verdict == Verdict::Keep) {
            selected.push_back(candidate.id);
        } else if (verdict == Verdict::Duplicate) {
            selected[slot] = std::max(selected[slot], candidate.id);
        } else if (keepPrunedConnections_) {
            pruned.push_back(candidate.id);
        }
    }
    const std::size_t refill = std::min(maxNeighbors - selected.size(), pruned.size());
    selected.insert(
        selected.end(), pruned.begin(), pruned.begin() + static_cast<std::ptrdiff_t>(refill));
    return selected;
}

}
