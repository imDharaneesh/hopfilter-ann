#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/distance/metric_registry.hpp>
#include <hopfilter/distance/metrics.hpp>

namespace hopfilter {
namespace {

template <typename Metric>
MetricRegistry::Factory factoryFor() {
    return [] { return std::make_unique<Metric>(); };
}

std::string joined(const std::vector<std::string>& names) {
    std::string result;
    for (const std::string& name : names) {
        result += result.empty() ? name : ", " + name;
    }
    return result;
}

}  // namespace

MetricRegistry MetricRegistry::withBuiltins() {
    MetricRegistry registry;
    registry.registerMetric(std::string{kL2MetricName}, factoryFor<L2Metric>());
    registry.registerMetric(std::string{kInnerProductMetricName}, factoryFor<InnerProductMetric>());
    registry.registerMetric(std::string{kCosineMetricName}, factoryFor<CosineMetric>());
    return registry;
}

void MetricRegistry::registerMetric(std::string name, Factory factory) {
    if (name.empty() || !factory) {
        throw InvalidArgumentError("metric name and factory must be non-empty");
    }
    if (factories_.contains(name)) {
        throw InvalidArgumentError("metric '" + name + "' is already registered");
    }
    factories_.emplace(std::move(name), std::move(factory));
}

std::unique_ptr<DistanceMetric> MetricRegistry::create(std::string_view name) const {
    const auto found = factories_.find(name);
    if (found == factories_.end()) {
        throw InvalidArgumentError("unknown metric '" + std::string{name} +
                                   "'; known metrics: " + joined(names()));
    }
    return found->second();
}

std::vector<std::string> MetricRegistry::names() const {
    std::vector<std::string> result;
    result.reserve(factories_.size());
    for (const auto& [name, factory] : factories_) {
        result.push_back(name);
    }
    return result;
}

}  // namespace hopfilter
