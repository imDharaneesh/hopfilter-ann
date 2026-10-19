#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <vector>

#include <hopfilter/common/types.hpp>
#include <hopfilter/common/visited_set.hpp>
#include <hopfilter/distance/distance_metric.hpp>
#include <hopfilter/graph/hnsw_graph.hpp>
#include <hopfilter/graph/layer_assigner.hpp>
#include <hopfilter/graph/neighbor_selector.hpp>
#include <hopfilter/graph/node_distances.hpp>
#include <hopfilter/graph/node_locks.hpp>
#include <hopfilter/index/index_config.hpp>
#include <hopfilter/index/interfaces.hpp>
#include <hopfilter/search/search_context.hpp>
#include <hopfilter/search/search_strategy.hpp>
#include <hopfilter/storage/vector_store.hpp>

namespace hopfilter {

struct HnswIndexParts {
    IndexConfig config;
    std::unique_ptr<DistanceMetric> metric;
    std::unique_ptr<VectorStore> vectors;
    std::unique_ptr<HnswGraph> graph;
    std::unique_ptr<LayerAssigner> layerAssigner;
    std::unique_ptr<NeighborSelector> neighborSelector;
    std::unique_ptr<SearchStrategy> searchStrategy;
};

class HnswIndex final : public Insertable, public Searchable {
public:
    explicit HnswIndex(HnswIndexParts parts);

    NodeId insert(std::span<const float> vector) override;

    NodeId insertBatch(std::span<const float> rows, std::size_t threads);

    [[nodiscard]] SearchResults search(std::span<const float> query,
                                       std::size_t k,
                                       const SearchRequest& request) const override;

    [[nodiscard]] std::size_t size() const noexcept override { return vectors_->size(); }
    [[nodiscard]] Dimension dimension() const noexcept override { return config_.dimension; }
    [[nodiscard]] std::size_t capacity() const noexcept { return config_.capacity; }

    [[nodiscard]] const IndexConfig& config() const noexcept { return config_; }
    [[nodiscard]] const VectorStore& vectors() const noexcept { return *vectors_; }
    [[nodiscard]] const HnswGraph& graph() const noexcept { return *graph_; }
    [[nodiscard]] const DistanceMetric& metric() const noexcept { return *metric_; }

private:
    struct EntrySnapshot {
        std::optional<NodeId> entryPoint;
        LevelIndex topLevel = 0;
    };

    void validateDimension(std::span<const float> vector) const;
    [[nodiscard]] std::vector<float> prepared(std::span<const float> vector) const;
    [[nodiscard]] NodeId allocate(std::span<const float> rows);
    [[nodiscard]] EntrySnapshot readEntry() const;
    void linkNode(NodeId id);
    void linkBelow(NodeId id, NodeId entryPoint, LevelIndex topLevel, const SearchContext& context);
    void connectBack(NodeId neighbor, NodeId newNode, LevelIndex level);
    void mergeNeighbors(NodeId node, LevelIndex level, std::span<const NodeId> additions);
    [[nodiscard]] std::vector<NodeId> shrunk(NodeId node,
                                             LevelIndex level,
                                             std::span<const NodeId> ids) const;

    IndexConfig config_;
    std::unique_ptr<DistanceMetric> metric_;
    std::unique_ptr<VectorStore> vectors_;
    std::unique_ptr<HnswGraph> graph_;
    std::unique_ptr<LayerAssigner> layerAssigner_;
    std::unique_ptr<NeighborSelector> neighborSelector_;
    std::unique_ptr<SearchStrategy> searchStrategy_;
    NodeDistances distances_;
    mutable VisitedSetPool visitedPool_;
    std::mutex allocationMutex_;
    mutable std::shared_mutex entryMutex_;
    NodeLocks nodeLocks_;
};

}
