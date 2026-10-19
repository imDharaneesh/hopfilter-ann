#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hopfilter/distance/metric_registry.hpp>
#include <hopfilter/distance/metrics.hpp>

namespace hopfilter {
namespace {

template <typename Metric>
MetricRegistry::Factory factoryFor() {
    return [] { return std::make_unique<Metric>(); };
}

}

MetricRegistry MetricRegistry::withBuiltins() {
    MetricRegistry registry;
    registry.registerMetric(std::string{kL2MetricName}, factoryFor<L2Metric>());
    registry.registerMetric(std::string{kInnerProductMetricName}, factoryFor<InnerProductMetric>());
    registry.registerMetric(std::string{kCosineMetricName}, factoryFor<CosineMetric>());
    return registry;
}

void MetricRegistry::registerMetric(std::string name, Factory factory) {
    factories_.add(std::move(name), std::move(factory));
}

std::unique_ptr<DistanceMetric> MetricRegistry::create(std::string_view name) const {
    return factories_.create(name);
}

std::vector<std::string> MetricRegistry::names() const {
    return factories_.names();
}

}
