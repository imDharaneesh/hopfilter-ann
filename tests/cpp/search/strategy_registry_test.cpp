#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/search/filtered_search_config.hpp>
#include <hopfilter/search/search_strategy.hpp>
#include <hopfilter/search/strategy_registry.hpp>

namespace hopfilter {
namespace {

SearchStrategyRegistry::Factory plainFactory() {
    return [] { return std::make_unique<UnfilteredStrategy>(); };
}

TEST(SearchStrategyRegistryTest, BuiltinsCoverEveryStrategy) {
    const auto registry = SearchStrategyRegistry::withBuiltins();

    EXPECT_EQ(registry.names(),
              (std::vector<std::string>{
                  "adaptive", "in_graph", "post_filter", "pre_filter", "unfiltered"}));
    for (const std::string& name : registry.names()) {
        EXPECT_NE(registry.create(name), nullptr) << name;
    }
}

TEST(SearchStrategyRegistryTest, UnknownNamesListTheKnownOnes) {
    const auto registry = SearchStrategyRegistry::withBuiltins();

    try {
        static_cast<void>(registry.create("brute"));
        FAIL() << "expected an unknown strategy to throw";
    } catch (const InvalidArgumentError& error) {
        EXPECT_EQ(std::string{error.what()},
                  "unknown search strategy 'brute'; known: adaptive, in_graph, post_filter, "
                  "pre_filter, unfiltered");
    }
}

TEST(SearchStrategyRegistryTest, CustomStrategiesCanBeAdded) {
    auto registry = SearchStrategyRegistry::withBuiltins();
    registry.registerStrategy("plain", plainFactory());

    EXPECT_NE(registry.create("plain"), nullptr);
}

TEST(SearchStrategyRegistryTest, DuplicateNamesAreRejected) {
    auto registry = SearchStrategyRegistry::withBuiltins();

    EXPECT_THROW(registry.registerStrategy("adaptive", plainFactory()), InvalidArgumentError);
}

TEST(SearchStrategyRegistryTest, RejectsInvalidConfig) {
    FilteredSearchConfig config;
    config.overFetchFactor = 0.0;

    EXPECT_THROW(static_cast<void>(SearchStrategyRegistry::withBuiltins(config)),
                 InvalidArgumentError);
}

}
}
