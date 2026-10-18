#pragma once

#include <algorithm>
#include <span>

#include <hopfilter/common/prefetch.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/distance/distance_metric.hpp>
#include <hopfilter/storage/vector_store.hpp>

namespace hopfilter {

/// Read-only view that measures distances between stored nodes and to query vectors.
/// Holds references only; the metric and store must outlive it.
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

    /// Starts loading the vector of ``id`` into cache ahead of a distance computation.
    void prefetch(NodeId id) const noexcept { prefetchForRead(vectors_->at(id).data()); }

    /// Whether two stored vectors are bit-for-bit equal; stops at the first difference.
    [[nodiscard]] bool identical(NodeId lhs, NodeId rhs) const noexcept {
        return std::ranges::equal(vectors_->at(lhs), vectors_->at(rhs));
    }

private:
    const DistanceMetric* metric_;
    const VectorStore* vectors_;
};

}
