#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hopfilter/search/adaptive_strategy.hpp>
#include <hopfilter/search/filtered_search_config.hpp>
#include <hopfilter/search/filtered_strategies.hpp>
#include <hopfilter/search/search_strategy.hpp>
#include <hopfilter/search/strategy_registry.hpp>

namespace hopfilter {

SearchStrategyRegistry SearchStrategyRegistry::withBuiltins(const FilteredSearchConfig& config) {
    validate(config);
    SearchStrategyRegistry registry;
    registry.registerStrategy(std::string{kUnfilteredStrategyName},
                              [] { return std::make_unique<UnfilteredStrategy>(); });
    registry.registerStrategy(std::string{kPostFilterStrategyName}, [config] {
        return std::make_unique<PostFilterStrategy>(config.overFetchFactor);
    });
    registry.registerStrategy(std::string{kPreFilterStrategyName},
                              [] { return std::make_unique<PreFilterStrategy>(); });
    registry.registerStrategy(std::string{kInGraphStrategyName},
                              [config] { return std::make_unique<InGraphStrategy>(config); });
    registry.registerStrategy(std::string{kAdaptiveStrategyName},
                              [config] { return AdaptiveStrategy::withDefaults(config); });
    return registry;
}

void SearchStrategyRegistry::registerStrategy(std::string name, Factory factory) {
    factories_.add(std::move(name), std::move(factory));
}

std::unique_ptr<SearchStrategy> SearchStrategyRegistry::create(std::string_view name) const {
    return factories_.create(name);
}

std::vector<std::string> SearchStrategyRegistry::names() const {
    return factories_.names();
}

}
