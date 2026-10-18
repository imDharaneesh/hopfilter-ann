#pragma once

#include <memory>
#include <optional>

#include <hopfilter/distance/distance_metric.hpp>
#include <hopfilter/distance/metric_registry.hpp>
#include <hopfilter/graph/hnsw_graph.hpp>
#include <hopfilter/graph/layer_assigner.hpp>
#include <hopfilter/graph/neighbor_selector.hpp>
#include <hopfilter/index/hnsw_index.hpp>
#include <hopfilter/index/index_config.hpp>
#include <hopfilter/search/search_strategy.hpp>
#include <hopfilter/storage/vector_store.hpp>

namespace hopfilter {

/// The only way to construct an HnswIndex: validates the configuration and fills in
/// a default for every collaborator that was not injected.
///
/// Defaults: the metric named by ``config.metricName`` from the registry, an
/// ExponentialLayerAssigner seeded with ``config.seed``, a HeuristicNeighborSelector,
/// an UnfilteredStrategy, and empty storage sized to ``config.capacity``.
class HnswIndexBuilder {
public:
    HnswIndexBuilder& withConfig(IndexConfig config);
    HnswIndexBuilder& withMetricRegistry(MetricRegistry registry);
    HnswIndexBuilder& withMetric(std::unique_ptr<DistanceMetric> metric);
    HnswIndexBuilder& withLayerAssigner(std::unique_ptr<LayerAssigner> layerAssigner);
    HnswIndexBuilder& withNeighborSelector(std::unique_ptr<NeighborSelector> neighborSelector);
    HnswIndexBuilder& withSearchStrategy(std::unique_ptr<SearchStrategy> searchStrategy);

    /// Restores previously built storage, e.g. from IndexSerializer. Its shape must
    /// match the configuration.
    HnswIndexBuilder& withStorage(std::unique_ptr<VectorStore> vectors,
                                  std::unique_ptr<HnswGraph> graph);

    /// Throws InvalidArgumentError naming the offending field if the configuration or
    /// injected parts are inconsistent. The builder is left empty afterwards.
    [[nodiscard]] std::unique_ptr<HnswIndex> build();

private:
    [[nodiscard]] std::unique_ptr<DistanceMetric> takeMetric(const IndexConfig& config);
    void fillDefaults(const IndexConfig& config);
    void validateStorage(const IndexConfig& config) const;

    std::optional<IndexConfig> config_;
    std::optional<MetricRegistry> registry_;
    std::unique_ptr<DistanceMetric> metric_;
    std::unique_ptr<LayerAssigner> layerAssigner_;
    std::unique_ptr<NeighborSelector> neighborSelector_;
    std::unique_ptr<SearchStrategy> searchStrategy_;
    std::unique_ptr<VectorStore> vectors_;
    std::unique_ptr<HnswGraph> graph_;
};

}  // namespace hopfilter
