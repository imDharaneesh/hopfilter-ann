#pragma once

#include <atomic>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include <hopfilter/common/types.hpp>

namespace hopfilter {

class HnswGraph {
public:
    HnswGraph(std::size_t capacity,
              std::size_t maxNeighborsLevelZero,
              std::size_t maxNeighborsUpper);

    NodeId addNode(LevelIndex level);

    [[nodiscard]] std::span<const NodeId> neighbors(NodeId id, LevelIndex level) const noexcept;

    void prefetchNeighbors(NodeId id, LevelIndex level) const noexcept;

    void setNeighbors(NodeId id, LevelIndex level, std::span<const NodeId> neighbors);

    void setEntryPoint(NodeId id);

    [[nodiscard]] std::optional<NodeId> entryPoint() const noexcept { return entryPoint_; }
    [[nodiscard]] LevelIndex maxLevel() const noexcept { return maxLevel_; }
    [[nodiscard]] LevelIndex levelOf(NodeId id) const noexcept { return levels_[id]; }
    [[nodiscard]] std::size_t size() const noexcept {
        return size_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

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
