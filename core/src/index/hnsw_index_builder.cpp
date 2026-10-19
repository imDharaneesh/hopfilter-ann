#include <memory>
#include <string>
#include <utility>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/distance/metric_registry.hpp>
#include <hopfilter/graph/hnsw_graph.hpp>
#include <hopfilter/graph/layer_assigner.hpp>
#include <hopfilter/graph/neighbor_selector.hpp>
#include <hopfilter/index/hnsw_index.hpp>
#include <hopfilter/index/hnsw_index_builder.hpp>
#include <hopfilter/index/index_config.hpp>
#include <hopfilter/search/adaptive_strategy.hpp>
#include <hopfilter/search/search_strategy.hpp>
#include <hopfilter/storage/vector_store.hpp>

namespace hopfilter {
namespace {

constexpr std::size_t kMinMaxNeighbors = 2;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw InvalidArgumentError(message);
    }
}

void validateConfig(const IndexConfig& config) {
    require(config.dimension > 0, "dimension must be positive");
    require(config.capacity > 0, "capacity must be positive");
    require(config.maxNeighbors >= kMinMaxNeighbors, "maxNeighbors must be at least 2");
    require(config.efConstruction >= config.maxNeighbors,
            "efConstruction must be at least maxNeighbors");
}

}

HnswIndexBuilder& HnswIndexBuilder::withConfig(IndexConfig config) {
    config_ = std::move(config);
    return *this;
}

HnswIndexBuilder& HnswIndexBuilder::withMetricRegistry(MetricRegistry registry) {
    registry_ = std::move(registry);
    return *this;
}

HnswIndexBuilder& HnswIndexBuilder::withMetric(std::unique_ptr<DistanceMetric> metric) {
    metric_ = std::move(metric);
    return *this;
}

HnswIndexBuilder& HnswIndexBuilder::withLayerAssigner(
    std::unique_ptr<LayerAssigner> layerAssigner) {
    layerAssigner_ = std::move(layerAssigner);
    return *this;
}

HnswIndexBuilder& HnswIndexBuilder::withNeighborSelector(
    std::unique_ptr<NeighborSelector> neighborSelector) {
    neighborSelector_ = std::move(neighborSelector);
    return *this;
}

HnswIndexBuilder& HnswIndexBuilder::withSearchStrategy(
    std::unique_ptr<SearchStrategy> searchStrategy) {
    searchStrategy_ = std::move(searchStrategy);
    return *this;
}

HnswIndexBuilder& HnswIndexBuilder::withStorage(std::unique_ptr<VectorStore> vectors,
                                                std::unique_ptr<HnswGraph> graph) {
    vectors_ = std::move(vectors);
    graph_ = std::move(graph);
    return *this;
}

std::unique_ptr<HnswIndex> HnswIndexBuilder::build() {
    if (!config_) {
        throw InvalidArgumentError("an IndexConfig is required");
    }
    IndexConfig config = std::move(*config_);
    config_.reset();
    validateConfig(config);
    validateStorage(config);
    auto metric = takeMetric(config);
    fillDefaults(config);
    HnswIndexParts parts{
        .config = std::move(config),
        .metric = std::move(metric),
        .vectors = std::move(vectors_),
        .graph = std::move(graph_),
        .layerAssigner = std::move(layerAssigner_),
        .neighborSelector = std::move(neighborSelector_),
        .searchStrategy = std::move(searchStrategy_),
    };
    return std::make_unique<HnswIndex>(std::move(parts));
}

void HnswIndexBuilder::fillDefaults(const IndexConfig& config) {
    if (!vectors_) {
        vectors_ = std::make_unique<VectorStore>(config.dimension, config.capacity);
        graph_ = std::make_unique<HnswGraph>(
            config.capacity, config.maxNeighborsLevelZero(), config.maxNeighbors);
    }
    if (!layerAssigner_) {
        layerAssigner_ =
            std::make_unique<ExponentialLayerAssigner>(config.maxNeighbors, config.seed);
    }
    if (!neighborSelector_) {
        neighborSelector_ = std::make_unique<HeuristicNeighborSelector>();
    }
    if (!searchStrategy_) {
        searchStrategy_ = AdaptiveStrategy::withDefaults();
    }
}

std::unique_ptr<DistanceMetric> HnswIndexBuilder::takeMetric(const IndexConfig& config) {
    if (!metric_) {
        const MetricRegistry registry = registry_ ? *registry_ : MetricRegistry::withBuiltins();
        return registry.create(config.metricName);
    }
    require(metric_->name() == config.metricName,
            "injected metric '" + std::string{metric_->name()} + "' does not match metricName '" +
                config.metricName + "'");
    return std::move(metric_);
}

void HnswIndexBuilder::validateStorage(const IndexConfig& config) const {
    require((vectors_ == nullptr) == (graph_ == nullptr),
            "vectors and graph must be restored together");
    if (!vectors_) {
        return;
    }
    require(vectors_->dimension() == config.dimension, "restored vectors have the wrong dimension");
    require(vectors_->capacity() == config.capacity && graph_->capacity() == config.capacity,
            "restored storage has the wrong capacity");
    require(graph_->size() == vectors_->size(), "restored graph and vectors differ in size");
    require(graph_->maxNeighbors(0) == config.maxNeighborsLevelZero() &&
                graph_->maxNeighbors(1) == config.maxNeighbors,
            "restored graph has the wrong degree bounds");
}

}
