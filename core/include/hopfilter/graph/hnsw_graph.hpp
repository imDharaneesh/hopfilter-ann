#pragma once

#include <atomic>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include <hopfilter/common/types.hpp>

namespace hopfilter {

/// Layered adjacency lists of an HNSW index: structure only, no vectors or distances.
///
/// Level 0 holds most edges and is read on every query, so it lives in one flat,
/// preallocated array with ``maxNeighborsLevelZero + 1`` slots per node (a count
/// followed by ids). Upper levels hold few nodes and use one small vector per node.
/// Node ids are dense: the n-th added node has id n. All per-node storage is allocated up
/// front, so adding a node never moves another node's data. `addNode` calls must be
/// serialized; reads and writes of different nodes' neighbor lists may run concurrently
/// as long as each node is guarded, e.g. with NodeLocks.
class HnswGraph {
public:
    /// Throws InvalidArgumentError if any argument is zero or ``capacity`` does not
    /// fit in a NodeId.
    HnswGraph(std::size_t capacity,
              std::size_t maxNeighborsLevelZero,
              std::size_t maxNeighborsUpper);

    /// Adds the next node with top layer ``level`` and returns its id.
    /// Throws IndexStateError when the graph is full.
    NodeId addNode(LevelIndex level);

    /// Neighbors of ``id`` on ``level``. ``level`` must not exceed ``levelOf(id)``.
    [[nodiscard]] std::span<const NodeId> neighbors(NodeId id, LevelIndex level) const noexcept;

    /// Starts loading the neighbor list of ``id`` on ``level`` into cache.
    void prefetchNeighbors(NodeId id, LevelIndex level) const noexcept;

    /// Replaces the neighbors of ``id`` on ``level``. Throws InvalidArgumentError if
    /// ``level`` is above the node's top layer or there are too many neighbors.
    void setNeighbors(NodeId id, LevelIndex level, std::span<const NodeId> neighbors);

    /// Makes ``id`` the search entry point; the graph's top layer becomes its level.
    void setEntryPoint(NodeId id);

    [[nodiscard]] std::optional<NodeId> entryPoint() const noexcept { return entryPoint_; }
    [[nodiscard]] LevelIndex maxLevel() const noexcept { return maxLevel_; }
    [[nodiscard]] LevelIndex levelOf(NodeId id) const noexcept { return levels_[id]; }
    [[nodiscard]] std::size_t size() const noexcept {
        return size_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    /// Degree bound on ``level``: ``maxNeighborsLevelZero`` on level 0, else ``maxNeighborsUpper``.
    [[nodiscard]] std::size_t maxNeighbors(LevelIndex level) const noexcept;

private:
    template <typename Self>
    [[nodiscard]] static auto slotOf(Self& graph, NodeId id, LevelIndex level) noexcept;

    std::size_t capacity_;
    std::size_t maxNeighborsLevelZero_;
    std::size_t maxNeighborsUpper_;
    std::vector<NodeId> levelZeroLinks_;
    std::vector<std::vector<NodeId>> upperLinks_;
    std::vector<LevelIndex> levels_;
    std::atomic<std::size_t> size_{0};
    std::optional<NodeId> entryPoint_;
    LevelIndex maxLevel_ = 0;
};

}
