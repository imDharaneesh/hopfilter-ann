#pragma once

#include <span>
#include <string_view>

#include <hopfilter/distance/distance_metric.hpp>

namespace hopfilter {

inline constexpr std::string_view kL2MetricName = "l2";
inline constexpr std::string_view kInnerProductMetricName = "ip";
inline constexpr std::string_view kCosineMetricName = "cosine";

/// Squared Euclidean distance. Monotonic in true L2, so rankings are identical.
class L2Metric final : public DistanceMetric {
public:
    [[nodiscard]] float distance(std::span<const float> lhs,
                                 std::span<const float> rhs) const noexcept override;
    [[nodiscard]] std::string_view name() const noexcept override;
};

/// ``1 - dot(lhs, rhs)``: larger inner products become smaller distances.
class InnerProductMetric final : public DistanceMetric {
public:
    [[nodiscard]] float distance(std::span<const float> lhs,
                                 std::span<const float> rhs) const noexcept override;
    [[nodiscard]] std::string_view name() const noexcept override;
};

/// ``1 - cosine(lhs, rhs)``. Vectors are L2-normalized once in ``prepare`` so each
/// distance call is a plain dot product. A zero vector stays zero.
class CosineMetric final : public DistanceMetric {
public:
    [[nodiscard]] float distance(std::span<const float> lhs,
                                 std::span<const float> rhs) const noexcept override;
    [[nodiscard]] std::string_view name() const noexcept override;
    void prepare(std::span<float> vector) const noexcept override;
};

}  // namespace hopfilter
