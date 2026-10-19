#pragma once

#include <algorithm>
#include <cstddef>
#include <span>

#include <hopfilter/common/prefetch.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/distance/distance_metric.hpp>
#include <hopfilter/storage/vector_store.hpp>

namespace hopfilter {

class NodeDistances {
public:
    NodeDistances(const DistanceMetric& metric, const VectorStore& vectors) noexcept
        : metric_(&metric), vectors_(&vectors) {}

    [[nodiscard]] float between(NodeId lhs, NodeId rhs) const noexcept {
        return metric_->distance(vectors_->at(lhs), vectors_->at(rhs));
    }

    [[nodiscard]] float toQuery(std::span<const float> query, NodeId id) const noexcept {
        return metric_->distance(query, vectors_->at(id));
    }

    void prefetch(NodeId id) const noexcept {
        const auto vector = vectors_->at(id);
        for (std::size_t offset = 0; offset < vector.size();
             offset += VectorStore::kFloatsPerAlignment) {
            prefetchForRead(vector.subspan(offset).data());
        }
    }

    [[nodiscard]] bool identical(NodeId lhs, NodeId rhs) const noexcept {
        return std::ranges::equal(vectors_->at(lhs), vectors_->at(rhs));
    }

private:
    const DistanceMetric* metric_;
    const VectorStore* vectors_;
};

}
