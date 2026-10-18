#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/graph/hnsw_graph.hpp>

namespace hopfilter {
namespace {

constexpr std::size_t kCountSlots = 1;

void validateShape(std::size_t capacity, std::size_t levelZero, std::size_t upper) {
    if (capacity == 0 || levelZero == 0 || upper == 0) {
        throw InvalidArgumentError("graph capacity and degree bounds must be positive");
    }
    if (capacity >= kInvalidNodeId) {
        throw InvalidArgumentError("graph capacity must be below " +
                                   std::to_string(kInvalidNodeId));
    }
}

}

template <typename Self>
auto HnswGraph::slotOf(Self& graph, NodeId id, LevelIndex level) noexcept {
    using Element = std::conditional_t<std::is_const_v<Self>, const NodeId, NodeId>;
    if (level == 0) {
        const std::size_t width = graph.maxNeighborsLevelZero_ + kCountSlots;
        return std::span<Element>{graph.levelZeroLinks_}.subspan(id * width, width);
    }
    const std::size_t width = graph.maxNeighborsUpper_ + kCountSlots;
    const std::size_t offset = static_cast<std::size_t>(level - 1) * width;
    return std::span<Element>{graph.upperLinks_[id]}.subspan(offset, width);
}

HnswGraph::HnswGraph(std::size_t capacity,
                     std::size_t maxNeighborsLevelZero,
                     std::size_t maxNeighborsUpper)
    : capacity_((validateShape(capacity, maxNeighborsLevelZero, maxNeighborsUpper), capacity)),
      maxNeighborsLevelZero_(maxNeighborsLevelZero),
      maxNeighborsUpper_(maxNeighborsUpper),
      levelZeroLinks_(capacity * (maxNeighborsLevelZero + kCountSlots), 0) {
    upperLinks_.reserve(capacity);
    levels_.reserve(capacity);
}

NodeId HnswGraph::addNode(LevelIndex level) {
    if (size() == capacity_) {
        throw IndexStateError("graph is full (capacity " + std::to_string(capacity_) + ")");
    }
    levels_.push_back(level);
    upperLinks_.emplace_back(static_cast<std::size_t>(level) * (maxNeighborsUpper_ + kCountSlots),
                             0);
    return static_cast<NodeId>(size() - 1);
}

std::span<const NodeId> HnswGraph::neighbors(NodeId id, LevelIndex level) const noexcept {
    const auto links = slotOf(*this, id, level);
    return links.subspan(kCountSlots, links.front());
}

void HnswGraph::setNeighbors(NodeId id, LevelIndex level, std::span<const NodeId> neighbors) {
    if (level > levelOf(id)) {
        throw InvalidArgumentError("node " + std::to_string(id) + " has no level " +
                                   std::to_string(level));
    }
    if (neighbors.size() > maxNeighbors(level)) {
        throw InvalidArgumentError("too many neighbors for level " + std::to_string(level));
    }
    const auto links = slotOf(*this, id, level);
    links.front() = static_cast<NodeId>(neighbors.size());
    std::ranges::copy(neighbors, links.begin() + kCountSlots);
}

void HnswGraph::setEntryPoint(NodeId id) {
    entryPoint_ = id;
    maxLevel_ = levelOf(id);
}

std::size_t HnswGraph::maxNeighbors(LevelIndex level) const noexcept {
    return level == 0 ? maxNeighborsLevelZero_ : maxNeighborsUpper_;
}

}
