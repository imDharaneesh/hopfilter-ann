#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include <hopfilter/common/types.hpp>
#include <hopfilter/distance/distance_metric.hpp>
#include <hopfilter/index/hnsw_index.hpp>
#include <hopfilter/index/hnsw_index_builder.hpp>
#include <hopfilter/index/index_config.hpp>

namespace hopfilter::testing {

/// Exact top-k ids of ``query`` over ``data`` under ``metric``, closest first.
inline std::vector<NodeId> bruteForceTopK(const std::vector<std::vector<float>>& data,
                                          std::span<const float> query,
                                          std::size_t k,
                                          const DistanceMetric& metric) {
    std::vector<float> preparedQuery(query.begin(), query.end());
    metric.prepare(preparedQuery);
    std::vector<Neighbor> all;
    all.reserve(data.size());
    for (std::size_t id = 0; id < data.size(); ++id) {
        std::vector<float> point = data[id];
        metric.prepare(point);
        all.push_back({static_cast<NodeId>(id), metric.distance(preparedQuery, point)});
    }
    const std::size_t count = std::min(k, all.size());
    std::ranges::partial_sort(all, all.begin() + static_cast<std::ptrdiff_t>(count), CloserFirst{});
    std::vector<NodeId> ids;
    for (std::size_t index = 0; index < count; ++index) {
        ids.push_back(all[index].id);
    }
    return ids;
}

/// Fraction of ``truth`` present in ``found``.
inline double recall(const SearchResults& found, const std::vector<NodeId>& truth) {
    std::unordered_set<NodeId> foundIds;
    for (const auto& neighbor : found) {
        foundIds.insert(neighbor.id);
    }
    const auto hits =
        std::ranges::count_if(truth, [&](NodeId id) { return foundIds.contains(id); });
    return truth.empty() ? 1.0 : static_cast<double>(hits) / static_cast<double>(truth.size());
}

/// Builds an index over ``data`` with default collaborators; capacity and dimension
/// are taken from ``data``.
inline std::unique_ptr<HnswIndex> buildIndex(const std::vector<std::vector<float>>& data,
                                             IndexConfig config) {
    config.dimension = data.front().size();
    config.capacity = data.size();
    auto index = HnswIndexBuilder{}.withConfig(std::move(config)).build();
    for (const auto& point : data) {
        static_cast<void>(index->insert(point));
    }
    return index;
}

}  // namespace hopfilter::testing
