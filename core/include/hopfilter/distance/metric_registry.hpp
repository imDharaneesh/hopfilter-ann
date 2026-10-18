#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hopfilter/distance/distance_metric.hpp>

namespace hopfilter {

/// Factory that builds metrics from configuration names. New metrics are added by
/// registering a factory; no existing class changes.
class MetricRegistry {
public:
    using Factory = std::function<std::unique_ptr<DistanceMetric>()>;

    /// Registry with "l2", "ip" and "cosine".
    [[nodiscard]] static MetricRegistry withBuiltins();

    /// Throws InvalidArgumentError if ``name`` is empty or already registered.
    void registerMetric(std::string name, Factory factory);

    /// Throws InvalidArgumentError naming the known metrics if ``name`` is unknown.
    [[nodiscard]] std::unique_ptr<DistanceMetric> create(std::string_view name) const;

    [[nodiscard]] std::vector<std::string> names() const;

private:
    using Entry = std::pair<std::string, Factory>;

    std::vector<Entry> factories_;
};

}
