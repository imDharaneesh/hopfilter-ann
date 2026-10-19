#include <algorithm>
#include <cstddef>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/parallel.hpp>
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
    return insertBatch(vector, 1);
}

NodeId HnswIndex::insertBatch(std::span<const float> rows, std::size_t threads) {
    const NodeId first = allocate(rows);
    const std::size_t count = rows.size() / config_.dimension;
    parallelFor(count, threads, [this, first](std::size_t offset) {
        linkNode(static_cast<NodeId>(first + offset));
    });
    return first;
}

SearchResults HnswIndex::search(std::span<const float> query,
                                std::size_t k,
                                const SearchRequest& request) const {
    return search(query, k, request, *searchStrategy_);
}

SearchResults HnswIndex::search(std::span<const float> query,
                                std::size_t k,
                                const SearchRequest& request,
                                const SearchStrategy& strategy) const {
    validateDimension(query);
    if (k == 0) {
        throw InvalidArgumentError("k must be positive");
    }
    const auto preparedQuery = prepared(query);
    const auto lease = visitedPool_.acquire();
    const SearchContext context{*graph_, distances_, lease.get()};
    return strategy.search(context, preparedQuery, k, request);
}

FilterPlan HnswIndex::planFor(std::span<const float> query,
                              const SearchRequest& request,
                              const AdaptiveStrategy& strategy) const {
    validateDimension(query);
    const auto preparedQuery = prepared(query);
    const auto lease = visitedPool_.acquire();
    const SearchContext context{*graph_, distances_, lease.get()};
    return strategy.planFor(context, preparedQuery, request);
}

void HnswIndex::setCategoricalAttribute(std::string name, CategoricalColumn codes) {
    validateColumnLength(name, codes.size());
    attributes_.setCategorical(std::move(name), std::move(codes));
}

void HnswIndex::setNumericAttribute(std::string name, NumericColumn values) {
    validateColumnLength(name, values.size());
    attributes_.setNumeric(std::move(name), std::move(values));
}

void HnswIndex::validateColumnLength(const std::string& name, std::size_t length) const {
    if (length > config_.capacity) {
        throw InvalidArgumentError("attribute '" + name + "' has " + std::to_string(length) +
                                   " rows but the index capacity is " +
                                   std::to_string(config_.capacity));
    }
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

NodeId HnswIndex::allocate(std::span<const float> rows) {
    if (rows.size() % config_.dimension != 0) {
        throw InvalidArgumentError("batch size " + std::to_string(rows.size()) +
                                   " is not a multiple of dimension " +
                                   std::to_string(config_.dimension));
    }
    const std::size_t count = rows.size() / config_.dimension;
    const std::scoped_lock lock{allocationMutex_};
    const std::size_t stored = vectors_->size();
    if (stored + count > config_.capacity) {
        throw IndexStateError("index is full: cannot add " + std::to_string(count) +
                              " vectors to " + std::to_string(stored) + " of capacity " +
                              std::to_string(config_.capacity));
    }
    for (std::size_t row = 0; row < count; ++row) {
        static_cast<void>(
            vectors_->append(prepared(rows.subspan(row * config_.dimension, config_.dimension))));
        static_cast<void>(graph_->addNode(layerAssigner_->nextLevel()));
    }
    return static_cast<NodeId>(stored);
}

HnswIndex::EntrySnapshot HnswIndex::readEntry() const {
    const std::shared_lock lock{entryMutex_};
    return {graph_->entryPoint(), graph_->maxLevel()};
}

void HnswIndex::linkNode(NodeId id) {
    const LevelIndex level = graph_->levelOf(id);
    std::unique_lock promotion{entryMutex_, std::defer_lock};
    auto snapshot = readEntry();
    if (!snapshot.entryPoint || level > snapshot.topLevel) {
        promotion.lock();
        snapshot = {graph_->entryPoint(), graph_->maxLevel()};
        if (!snapshot.entryPoint) {
            graph_->setEntryPoint(id);
            return;
        }
    }
    const auto lease = visitedPool_.acquire();
    const SearchContext context{*graph_, distances_, lease.get(), &nodeLocks_};
    linkBelow(id, *snapshot.entryPoint, snapshot.topLevel, context);
    if (promotion.owns_lock() && level > graph_->maxLevel()) {
        graph_->setEntryPoint(id);
    }
}

void HnswIndex::linkBelow(NodeId id,
                          NodeId entryPoint,
                          LevelIndex topLevel,
                          const SearchContext& context) {
    const auto vector = vectors_->at(id);
    const LevelIndex level = graph_->levelOf(id);
    const Neighbor entry{entryPoint, distances_.toQuery(vector, entryPoint)};
    std::vector<Neighbor> entries{greedyDescend(context, vector, entry, topLevel, level)};
    for (int current = std::min(level, topLevel); current >= 0; --current) {
        const auto layer = static_cast<LevelIndex>(current);
        auto candidates = searchLayer(context, vector, entries, config_.efConstruction, layer);
        std::erase_if(candidates, [id](const Neighbor& candidate) { return candidate.id == id; });
        const auto selected =
            neighborSelector_->select(candidates, config_.maxNeighbors, distances_);
        {
            const auto guard = nodeLocks_.lock(id);
            mergeNeighbors(id, layer, selected);
        }
        for (const NodeId neighbor : selected) {
            connectBack(neighbor, id, layer);
        }
        entries = candidates.empty() ? entries : std::move(candidates);
    }
}

void HnswIndex::connectBack(NodeId neighbor, NodeId newNode, LevelIndex level) {
    const auto guard = nodeLocks_.lock(neighbor);
    mergeNeighbors(neighbor, level, {&newNode, 1});
}

void HnswIndex::mergeNeighbors(NodeId node, LevelIndex level, std::span<const NodeId> additions) {
    const auto existing = graph_->neighbors(node, level);
    std::vector<NodeId> merged(existing.begin(), existing.end());
    for (const NodeId addition : additions) {
        if (std::ranges::find(merged, addition) == merged.end()) {
            merged.push_back(addition);
        }
    }
    if (merged.size() > graph_->maxNeighbors(level)) {
        merged = shrunk(node, level, merged);
    }
    graph_->setNeighbors(node, level, merged);
}

std::vector<NodeId> HnswIndex::shrunk(NodeId node,
                                      LevelIndex level,
                                      std::span<const NodeId> ids) const {
    std::vector<Neighbor> candidates;
    candidates.reserve(ids.size());
    for (const NodeId current : ids) {
        candidates.push_back({current, distances_.between(node, current)});
    }
    std::ranges::sort(candidates, CloserFirst{});
    return neighborSelector_->select(candidates, graph_->maxNeighbors(level), distances_);
}

}
