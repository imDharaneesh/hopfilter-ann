#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hopfilter/distance/distance_metric.hpp>

namespace hopfilter {

class MetricRegistry {
public:
    using Factory = std::function<std::unique_ptr<DistanceMetric>()>;

    [[nodiscard]] static MetricRegistry withBuiltins();

    void registerMetric(std::string name, Factory factory);

    [[nodiscard]] std::unique_ptr<DistanceMetric> create(std::string_view name) const;

    [[nodiscard]] std::vector<std::string> names() const;

private:
    using Entry = std::pair<std::string, Factory>;

    std::vector<Entry> factories_;
};

}
