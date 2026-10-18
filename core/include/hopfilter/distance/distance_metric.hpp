#pragma once

#include <span>
#include <string_view>

namespace hopfilter {

/// Strategy for comparing two vectors. Smaller distances always mean closer, so search
/// code never branches on the metric.
///
/// Implementations are stateless and safe to call from many threads at once.
class DistanceMetric {
public:
    virtual ~DistanceMetric() = default;

    /// Distance between two vectors of equal size that have both been prepared.
    [[nodiscard]] virtual float distance(std::span<const float> lhs,
                                         std::span<const float> rhs) const noexcept = 0;

    /// Registry name, e.g. "l2".
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /// Transforms a vector in place before it is stored or used as a query.
    virtual void prepare(std::span<float> vector) const noexcept;
};

}  // namespace hopfilter
