#include <algorithm>
#include <cstddef>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/index/hnsw_index.hpp>
#include <hopfilter/search/search_context.hpp>
#include <hopfilter/search/search_layer.hpp>

namespace hopfilter {

HnswIndex::HnswIndex(HnswIndexParts parts)
    : config_(std::move(parts.config)),
      metric_(std::move(parts.metric)),
      vectors_(std::move(parts.vectors)),
      graph_(std::move(parts.graph)),
      layerAssigner_(std::move(parts.layerAssigner)),
      neighborSelector_(std::move(parts.neighborSelector)),
      searchStrategy_(std::move(parts.searchStrategy)),
      distances_(*metric_, *vectors_),
      visitedPool_(config_.capacity) {}

NodeId HnswIndex::insert(std::span<const float> vector) {
    validateDimension(vector);
    const auto stored = prepared(vector);
    const std::scoped_lock lock{insertMutex_};
    const NodeId id = vectors_->append(stored);
    const LevelIndex level = layerAssigner_->nextLevel();
    static_cast<void>(graph_->addNode(level));
    const auto entryPoint = graph_->entryPoint();
    if (!entryPoint) {
        graph_->setEntryPoint(id);
        return id;
    }
    link(id, level, *entryPoint);
    if (level > graph_->maxLevel()) {
        graph_->setEntryPoint(id);
    }
    return id;
}

SearchResults HnswIndex::search(std::span<const float> query,
                                std::size_t k,
                                const SearchRequest& request) const {
    validateDimension(query);
    if (k == 0) {
        throw InvalidArgumentError("k must be positive");
    }
    const auto preparedQuery = prepared(query);
    const auto lease = visitedPool_.acquire();
    const SearchContext context{*graph_, distances_, lease.get()};
    return searchStrategy_->search(context, preparedQuery, k, request);
}

void HnswIndex::validateDimension(std::span<const float> vector) const {
    if (vector.size() != config_.dimension) {
        throw InvalidArgumentError("expected dimension " + std::to_string(config_.dimension) +
                                   ", got " + std::to_string(vector.size()));
    }
}

std::vector<float> HnswIndex::prepared(std::span<const float> vector) const {
    std::vector<float> copy(vector.begin(), vector.end());
    metric_->prepare(copy);
    return copy;
}

void HnswIndex::link(NodeId id, LevelIndex level, NodeId entryPoint) {
    const auto vector = vectors_->at(id);
    const auto lease = visitedPool_.acquire();
    const SearchContext context{*graph_, distances_, lease.get()};
    const LevelIndex topLevel = graph_->maxLevel();
    const Neighbor entry{entryPoint, distances_.toQuery(vector, entryPoint)};
    std::vector<Neighbor> entries{greedyDescend(context, vector, entry, topLevel, level)};
    for (int current = std::min(level, topLevel); current >= 0; --current) {
        const auto layer = static_cast<LevelIndex>(current);
        auto candidates = searchLayer(context, vector, entries, config_.efConstruction, layer);
        const auto selected =
            neighborSelector_->select(candidates, config_.maxNeighbors, distances_);
        graph_->setNeighbors(id, layer, selected);
        for (const NodeId neighbor : selected) {
            connectBack(neighbor, id, layer);
        }
        entries = std::move(candidates);
    }
}

void HnswIndex::connectBack(NodeId neighbor, NodeId newNode, LevelIndex level) {
    const auto existing = graph_->neighbors(neighbor, level);
    std::vector<NodeId> updated(existing.begin(), existing.end());
    if (updated.size() < graph_->maxNeighbors(level)) {
        updated.push_back(newNode);
        graph_->setNeighbors(neighbor, level, updated);
        return;
    }
    std::vector<Neighbor> candidates;
    candidates.reserve(updated.size() + 1);
    for (const NodeId current : updated) {
        candidates.push_back({current, distances_.between(neighbor, current)});
    }
    candidates.push_back({newNode, distances_.between(neighbor, newNode)});
    std::ranges::sort(candidates, CloserFirst{});
    const auto kept =
        neighborSelector_->select(candidates, graph_->maxNeighbors(level), distances_);
    graph_->setNeighbors(neighbor, level, kept);
}

}  // namespace hopfilter
