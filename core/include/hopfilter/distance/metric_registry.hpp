#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <hopfilter/common/named_factories.hpp>
#include <hopfilter/distance/distance_metric.hpp>

namespace hopfilter {

class MetricRegistry {
public:
    using Factory = NamedFactories<DistanceMetric>::Factory;

    [[nodiscard]] static MetricRegistry withBuiltins();

    void registerMetric(std::string name, Factory factory);

    [[nodiscard]] std::unique_ptr<DistanceMetric> create(std::string_view name) const;

    [[nodiscard]] std::vector<std::string> names() const;

private:
    NamedFactories<DistanceMetric> factories_{"metric"};
};

}
