#include <cmath>
#include <span>
#include <string_view>

#include <hopfilter/distance/distance_metric.hpp>
#include <hopfilter/distance/kernels.hpp>
#include <hopfilter/distance/metrics.hpp>

namespace hopfilter {

void DistanceMetric::prepare(std::span<float> /*vector*/) const noexcept {}

float L2Metric::distance(std::span<const float> lhs, std::span<const float> rhs) const noexcept {
    return squaredL2(lhs, rhs);
}

std::string_view L2Metric::name() const noexcept {
    return kL2MetricName;
}

float InnerProductMetric::distance(std::span<const float> lhs,
                                   std::span<const float> rhs) const noexcept {
    return 1.0F - dotProduct(lhs, rhs);
}

std::string_view InnerProductMetric::name() const noexcept {
    return kInnerProductMetricName;
}

float CosineMetric::distance(std::span<const float> lhs,
                             std::span<const float> rhs) const noexcept {
    return 1.0F - dotProduct(lhs, rhs);
}

std::string_view CosineMetric::name() const noexcept {
    return kCosineMetricName;
}

void CosineMetric::prepare(std::span<float> vector) const noexcept {
    const float norm = std::sqrt(dotProduct(vector, vector));
    if (norm == 0.0F) {
        return;
    }
    for (float& component : vector) {
        component /= norm;
    }
}

}  // namespace hopfilter
