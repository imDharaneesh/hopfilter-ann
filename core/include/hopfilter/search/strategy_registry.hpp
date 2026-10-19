#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <hopfilter/common/named_factories.hpp>
#include <hopfilter/search/filtered_search_config.hpp>
#include <hopfilter/search/search_strategy.hpp>

namespace hopfilter {

inline constexpr std::string_view kUnfilteredStrategyName = "unfiltered";
inline constexpr std::string_view kPostFilterStrategyName = "post_filter";
inline constexpr std::string_view kPreFilterStrategyName = "pre_filter";
inline constexpr std::string_view kInGraphStrategyName = "in_graph";
inline constexpr std::string_view kAdaptiveStrategyName = "adaptive";

class SearchStrategyRegistry {
public:
    using Factory = NamedFactories<SearchStrategy>::Factory;

    [[nodiscard]] static SearchStrategyRegistry withBuiltins(
        const FilteredSearchConfig& config = {});

    void registerStrategy(std::string name, Factory factory);

    [[nodiscard]] std::unique_ptr<SearchStrategy> create(std::string_view name) const;

    [[nodiscard]] std::vector<std::string> names() const;

private:
    NamedFactories<SearchStrategy> factories_{"search strategy"};
};

}
