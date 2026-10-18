#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include <hopfilter/common/types.hpp>
#include <hopfilter/common/visited_set.hpp>
#include <hopfilter/distance/distance_metric.hpp>
#include <hopfilter/graph/hnsw_graph.hpp>
#include <hopfilter/graph/layer_assigner.hpp>
#include <hopfilter/graph/neighbor_selector.hpp>
#include <hopfilter/graph/node_distances.hpp>
#include <hopfilter/index/index_config.hpp>
#include <hopfilter/index/interfaces.hpp>
#include <hopfilter/search/search_strategy.hpp>
#include <hopfilter/storage/vector_store.hpp>

namespace hopfilter {

/// Collaborators of an HnswIndex, assembled and validated by HnswIndexBuilder.
struct HnswIndexParts {
    IndexConfig config;
    std::unique_ptr<DistanceMetric> metric;
    std::unique_ptr<VectorStore> vectors;
    std::unique_ptr<HnswGraph> graph;
    std::unique_ptr<LayerAssigner> layerAssigner;
    std::unique_ptr<NeighborSelector> neighborSelector;
    std::unique_ptr<SearchStrategy> searchStrategy;
};

/// Hierarchical Navigable Small World index (Malkov and Yashunin, 2018).
///
/// Orchestrates insertion and search through injected collaborators; create it with
/// HnswIndexBuilder. Thread safety: any number of concurrent ``search`` calls are
/// safe; ``insert`` calls are serialized internally; ``search`` concurrent with
/// ``insert`` is not supported.
class HnswIndex final : public Insertable, public Searchable {
public:
    explicit HnswIndex(HnswIndexParts parts);

    /// Throws InvalidArgumentError on a dimension mismatch and IndexStateError when full.
    NodeId insert(std::span<const float> vector) override;

    /// Throws InvalidArgumentError on a dimension mismatch or ``k == 0``. Returns fewer
    /// than ``k`` neighbors only when the index holds fewer than ``k`` vectors.
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
    void validateDimension(std::span<const float> vector) const;
    [[nodiscard]] std::vector<float> prepared(std::span<const float> vector) const;
    void link(NodeId id, LevelIndex level, NodeId entryPoint);
    void connectBack(NodeId neighbor, NodeId newNode, LevelIndex level);

    IndexConfig config_;
    std::unique_ptr<DistanceMetric> metric_;
    std::unique_ptr<VectorStore> vectors_;
    std::unique_ptr<HnswGraph> graph_;
    std::unique_ptr<LayerAssigner> layerAssigner_;
    std::unique_ptr<NeighborSelector> neighborSelector_;
    std::unique_ptr<SearchStrategy> searchStrategy_;
    NodeDistances distances_;
    mutable VisitedSetPool visitedPool_;
    std::mutex insertMutex_;
};

}  // namespace hopfilter
